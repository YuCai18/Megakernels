#pragma once

#include "llama.cuh"

// matrix_pipeline: Multi-row matrix multiply pipeline for the encoder/prefill
// phase. Unlike matvec_pipeline (single-vector decode), this processes
// multiple token rows through Tensor Core mma_ABt, iterating over row blocks.

template <typename Config, typename Globals, typename parsed_instruction,
          typename pipeline_specifics, int TOKEN_STEP = 16>
struct matrix_pipeline {
    static constexpr int INPUT_PIPELINE_STAGES = 3;
    static constexpr int OUTPUT_PIPELINE_STAGES = 2;
    static constexpr int STAGE_PAGES = 4;
    static constexpr int ACTIVATION_PAGE = 0;
    static constexpr int WEIGHTS_START_PAGE = 1;

    static constexpr int REDUCTION_DIM_PER_WARP =
        Globals::hidden_dim / Config::NUM_CONSUMER_WARPS;

    static constexpr int SEM_COUNT =
        1 + (INPUT_PIPELINE_STAGES + OUTPUT_PIPELINE_STAGES) * 2;

    static constexpr int SCRATCH_BYTES_PER_WARP = TOKEN_STEP * sizeof(float);
    static constexpr int SCRATCH_BYTES_PER_STAGE =
        SCRATCH_BYTES_PER_WARP * Config::NUM_CONSUMER_WARPS;

    __device__ static inline int get_activation_page(megakernel::state<Config> &s) {
        return s.pid(ACTIVATION_PAGE);
    }

    __device__ static inline int get_weight_page(megakernel::state<Config> &s,
                                                  int stage, int offset) {
        return s.pid(WEIGHTS_START_PAGE + stage * STAGE_PAGES + offset);
    }

    __device__ static inline kittens::semaphore &activations_arrived(megakernel::state<Config> &s) {
        return s.semaphores()[0];
    }
    __device__ static inline kittens::semaphore &weights_arrived(megakernel::state<Config> &s, int stage) {
        return s.semaphores()[1 + stage];
    }
    __device__ static inline kittens::semaphore &weights_finished(megakernel::state<Config> &s, int stage) {
        return s.semaphores()[1 + INPUT_PIPELINE_STAGES + stage];
    }
    __device__ static inline kittens::semaphore &outputs_arrived(megakernel::state<Config> &s, int stage) {
        return s.semaphores()[1 + 2 * INPUT_PIPELINE_STAGES + stage];
    }
    __device__ static inline kittens::semaphore &outputs_finished(megakernel::state<Config> &s, int stage) {
        return s.semaphores()[1 + 2 * INPUT_PIPELINE_STAGES + OUTPUT_PIPELINE_STAGES + stage];
    }

    __device__ static inline uint8_t *get_output_start(megakernel::state<Config> &s,
                                                        int stage) {
        return (uint8_t *)s.scratch() + (stage * SCRATCH_BYTES_PER_STAGE);
    }

    __device__ static inline int
    release_lid(const Globals &g, typename Config::instruction_t &instruction,
                int &query) {
        return query; // identity; ops can override
    }

    __device__ static inline int init_semaphores(megakernel::state<Config> &s) {
        init_semaphore(activations_arrived(s), 1);
        for (int i = 0; i < INPUT_PIPELINE_STAGES; i++) {
            init_semaphore(weights_arrived(s, i), 1);
            init_semaphore(weights_finished(s, i), Config::NUM_CONSUMER_WARPS);
        }
        for (int i = 0; i < OUTPUT_PIPELINE_STAGES; i++) {
            init_semaphore(outputs_arrived(s, i), Config::NUM_CONSUMER_WARPS);
            init_semaphore(outputs_finished(s, i), 1);
        }
        return SEM_COUNT;
    }

    // Loader: streams weight tiles from global memory into shared memory pages
    __device__ static inline void loader_loop(megakernel::state<Config> &s,
                                              const Globals &g, int layer_idx) {
        parsed_instruction inst{s};

        auto needed_pages =
            1 + min(inst.iters, INPUT_PIPELINE_STAGES) * STAGE_PAGES;

        if (kittens::laneid() == 0) {
            int input_stage = 0;
            for (int iter = 0; iter < inst.iters; iter++) {
                kittens::wait(weights_finished(s, input_stage),
                     (iter % (2 * INPUT_PIPELINE_STAGES)) < INPUT_PIPELINE_STAGES);

                auto &sem = weights_arrived(s, input_stage);
                kittens::tma::expect_bytes(sem, sizeof(kittens::bf16) * Globals::hidden_dim * TOKEN_STEP);

#pragma unroll
                for (int i = 0; i < STAGE_PAGES; i++) {
                    int weight_page = get_weight_page(s, input_stage, i);
                    if (iter < INPUT_PIPELINE_STAGES) {
                        s.wait_page_ready(weight_page);
                    }
                    auto &weight_chunk = reinterpret_cast<kittens::st_bf<TOKEN_STEP, Globals::hidden_dim / STAGE_PAGES> &>(
                        s.pages[weight_page]);

                    pipeline_specifics::load_iter(s, g, inst, iter, i,
                                                  weight_chunk, sem);
                }
                input_stage = (input_stage + 1) % INPUT_PIPELINE_STAGES;
            }
        } else if (kittens::laneid() >= needed_pages && kittens::laneid() < Config::NUM_PAGES) {
            auto pid = s.pid(kittens::laneid());
            s.wait_page_ready(pid);
            s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
        }
    }

    // Consumer: matrix multiply via Tensor Core mma_ABt, row-step iteration
    __device__ static inline void consumer_loop(megakernel::state<Config> &s,
                                                 const Globals &g) {
        parsed_instruction inst{s};

        constexpr int WARPS_PER_PAGE = Config::NUM_CONSUMER_WARPS / STAGE_PAGES;
        int page_index = kittens::warpid() / WARPS_PER_PAGE;

        int input_stage = 0, output_stage = 0;
        for (int i = 0; i < inst.iters; i++) {
            int weight_page = get_weight_page(s, input_stage, page_index);
            kittens::wait(weights_arrived(s, input_stage),
                 (i % (2 * INPUT_PIPELINE_STAGES)) >= INPUT_PIPELINE_STAGES);
            kittens::wait(outputs_finished(s, output_stage),
                 (i % (2 * OUTPUT_PIPELINE_STAGES)) < OUTPUT_PIPELINE_STAGES);

            auto &weights_smem = reinterpret_cast<
                kittens::st_bf<TOKEN_STEP, REDUCTION_DIM_PER_WARP> *>(
                s.pages[weight_page].ptr())[kittens::warpid() % WARPS_PER_PAGE];

            pipeline_specifics::compute(s, g, inst, i, input_stage, output_stage,
                                        weights_smem);

            kittens::warp::sync();
            kittens::warp::arrive(outputs_arrived(s, output_stage));
            kittens::warp::arrive(weights_finished(s, input_stage));

            if (i >= inst.iters - INPUT_PIPELINE_STAGES) {
#pragma unroll
                for (int j = 0; j < STAGE_PAGES; j++) {
                    s.warp_finish_page(get_weight_page(s, input_stage, j), 1);
                }
            }

            input_stage = (input_stage + 1) % INPUT_PIPELINE_STAGES;
            output_stage = (output_stage + 1) % OUTPUT_PIPELINE_STAGES;
        }
    }

    // Storer: writes output from scratch to global memory
    __device__ static inline void storer_loop(megakernel::state<Config> &s,
                                              const Globals &g) {
        parsed_instruction inst{s};

        int output_stage = 0;
        for (int i = 0; i < inst.iters; i++) {
            kittens::wait(outputs_arrived(s, output_stage),
                 (i % (2 * OUTPUT_PIPELINE_STAGES)) >= OUTPUT_PIPELINE_STAGES);

            pipeline_specifics::store(s, g, inst, i, output_stage);

            kittens::warp::arrive(outputs_finished(s, output_stage));
            output_stage = (output_stage + 1) % OUTPUT_PIPELINE_STAGES;
        }
    }
};

// rms_matrix_pipeline: extends matrix_pipeline with RMS norm before matmul
template <typename Config, typename Globals, typename parsed_instruction,
          typename pipeline_specifics, auto ActPtr, auto RmsPtr,
          int TOKEN_STEP = 16>
struct rms_matrix_pipeline
    : public matrix_pipeline<Config, Globals, parsed_instruction,
                              pipeline_specifics, TOKEN_STEP> {
    using pipeline = matrix_pipeline<Config, Globals, parsed_instruction,
                                      pipeline_specifics, TOKEN_STEP>;

    static constexpr int REDUCTION_DIM_PER_WARP =
        Globals::hidden_dim / Config::NUM_CONSUMER_WARPS;

    static constexpr int SEM_COUNT = 1 + pipeline::SEM_COUNT;

    __device__ static inline kittens::semaphore &rms_scale_arrived(megakernel::state<Config> &s) {
        return s.semaphores()[pipeline::SEM_COUNT];
    }

    __device__ static inline kittens::sv_bf<Globals::hidden_dim> &
    get_rms_scale(megakernel::state<Config> &s) {
        return *reinterpret_cast<kittens::sv_bf<Globals::hidden_dim> *>(
            s.pages[pipeline::get_activation_page(s)].ptr(
                sizeof(kittens::sv_bf<Globals::hidden_dim>)));
    }

    __device__ static inline int init_semaphores(megakernel::state<Config> &s) {
        pipeline::init_semaphores(s);
        init_semaphore(rms_scale_arrived(s), 1);
        return SEM_COUNT;
    }

    __device__ static inline void loader_loop(megakernel::state<Config> &s,
                                              const Globals &g, int layer_idx) {
        if (kittens::laneid() == 0) {
            int activation_page = pipeline::get_activation_page(s);
            s.wait_page_ready(activation_page);

            auto &rms_scale = get_rms_scale(s);
            auto &sem = rms_scale_arrived(s);
            kittens::tma::expect(sem, rms_scale);
            kittens::tma::load_async<kittens::cache_policy::EVICT_LAST>(
                rms_scale, g.*RmsPtr, {layer_idx, 0}, sem);
        }
        pipeline::loader_loop(s, g, layer_idx);
    }

    __device__ static inline void launcher_loop(megakernel::state<Config> &s,
                                                 const Globals &g) {
        if (kittens::laneid() == 0) {
#ifdef KITTENS_BLACKWELL
            s.wait_tensor_ready();
            arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);
#endif
        }
    }

    __device__ static inline void consumer_loop(megakernel::state<Config> &s,
                                                 const Globals &g) {
        using sv_t = kittens::sv_bf<REDUCTION_DIM_PER_WARP>;
        auto &rms_scale_smem =
            reinterpret_cast<sv_t *>(&get_rms_scale(s))[kittens::warpid()];
        auto &activations_smem =
            reinterpret_cast<sv_t *>(&pipeline::get_activations(s))[kittens::warpid()];

        if (kittens::laneid() == 0 && kittens::warpid() == 0) {
            parsed_instruction inst{s};
            int activation_page = pipeline::get_activation_page(s);
            s.wait_page_ready(activation_page);

            s.record(megakernel::TEVENT_AT_GMEM_WAIT);
            pipeline_specifics::gmem_wait(g, s);
            s.record(megakernel::TEVENT_DONE_GMEM_WAIT);
        }
        kittens::group<Config::NUM_CONSUMER_WARPS>::sync(3);

        // Load activation rows and apply RMS norm before matrix multiply
        kittens::warp::load(activations_smem, g.*ActPtr, {kittens::warpid()});
        auto activation_page = pipeline::get_activation_page(s);
        kittens::wait(rms_scale_arrived(s), 0);

        auto activations_vec = rms_norm<Config>(
            rms_scale_smem, activations_smem, g.rms_norm_eps,
            pipeline::get_output_start(s, pipeline::OUTPUT_PIPELINE_STAGES));

        kittens::warp::sync();
        s.warp_finish_page(activation_page, 1);

        // Delegate actual matrix multiply to base pipeline
        pipeline::consumer_loop(s, g);
    }
};
