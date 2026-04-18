#pragma once

#include "llama.cuh"

// quantized_pipeline: Extends matvec_pipeline with quantization support.
//
// QuantMode template parameter:
//   0 = bf16 weights, bf16 activations (no quantization)
//   1 = INT4 weights with group scales, dequantized to bf16 before matmul
//   2 = INT4 weights dequantized to FP8, FP8 activations -> Tensor Core mma_ABt

template <int QuantMode> struct quant_traits {
    static constexpr bool has_quant = (QuantMode > 0);
    static constexpr bool has_fp8_act = (QuantMode == 2);
    static constexpr int scale_pipeline_stages = has_quant ? 3 : 0;
};

// INT4 -> bf16 dequantization utility
template<int group_size = 128>
__device__ static inline void dequant_int4_to_bf16(
    kittens::st_bf<16, 512> &output,
    const uint32_t *packed,
    const kittens::sv_bf<512 / group_size> &scales)
{
    constexpr int groups = 512 / group_size;
    for (int r = kittens::laneid() / 8; r < 16; r += 4) {
        for (int c = kittens::laneid() % 8; c < 512; c += 8) {
            int gi = c / group_size;
            uint16_t pv = reinterpret_cast<const uint16_t *>(packed)[r * 256 + c / 2];
            int lo = pv & 0xF, hi = (pv >> 4) & 0xF;
            float sf = __bfloat162float(scales[gi]);
            reinterpret_cast<kittens::bf16 *>(&output)[r * 512 + c] =
                __float2bfloat16((float(lo) - 8.f) * sf);
            reinterpret_cast<kittens::bf16 *>(&output)[r * 512 + c + 1] =
                __float2bfloat16((float(hi) - 8.f) * sf);
        }
    }
    kittens::warp::sync();
}

// quantized_matvec_pipeline: matvec pipeline + quantization scale pipeline
template <typename Config, typename Globals, typename parsed_instruction,
          typename pipeline_specifics, auto ActPtr, auto RmsPtr,
          int QuantMode = 0>
struct quantized_matvec_pipeline {
    using traits = quant_traits<QuantMode>;

    static constexpr int INPUT_PIPELINE_STAGES = 3;
    static constexpr int OUTPUT_PIPELINE_STAGES = 3;
    static constexpr int STAGE_PAGES = 4;
    static constexpr int REDUCTION_DIM_PER_WARP =
        Globals::hidden_dim / Config::NUM_CONSUMER_WARPS;

    // Extra semaphores for quantization scale pipeline
    static constexpr int QUANT_SEM_COUNT =
        traits::has_quant ? (traits::scale_pipeline_stages * 2) : 0;

    static constexpr int SEM_COUNT =
        1 + (INPUT_PIPELINE_STAGES + OUTPUT_PIPELINE_STAGES) * 2 + QUANT_SEM_COUNT + 1;

    static constexpr int SCRATCH_BYTES_PER_WARP = 16 * sizeof(float);
    static constexpr int SCRATCH_BYTES_PER_STAGE =
        SCRATCH_BYTES_PER_WARP * Config::NUM_CONSUMER_WARPS;

    static constexpr int ACTIVATION_PAGE = 0;
    static constexpr int WEIGHTS_START_PAGE = 1;

    __device__ static inline int get_activation_page(megakernel::state<Config> &s) {
        return s.pid(ACTIVATION_PAGE);
    }
    __device__ static inline int get_weight_page(megakernel::state<Config> &s, int stage, int offset) {
        return s.pid(WEIGHTS_START_PAGE + stage * STAGE_PAGES + offset);
    }

    // Base semaphores
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
    // Quantization scale semaphores
    __device__ static inline kittens::semaphore &scales_arrived(megakernel::state<Config> &s, int stage) {
        return s.semaphores()[1 + 2 * (INPUT_PIPELINE_STAGES + OUTPUT_PIPELINE_STAGES) + stage];
    }
    __device__ static inline kittens::semaphore &scales_finished(megakernel::state<Config> &s, int stage) {
        return s.semaphores()[1 + 2 * (INPUT_PIPELINE_STAGES + OUTPUT_PIPELINE_STAGES) +
                              traits::scale_pipeline_stages + stage];
    }
    // RMS scale
    __device__ static inline kittens::semaphore &rms_scale_arrived(megakernel::state<Config> &s) {
        return s.semaphores()[SEM_COUNT - 1];
    }

    __device__ static inline kittens::sv_bf<Globals::hidden_dim> &
    get_activations(megakernel::state<Config> &s) {
        return *reinterpret_cast<kittens::sv_bf<Globals::hidden_dim> *>(
            s.pages[get_activation_page(s)].ptr());
    }

    __device__ static inline kittens::sv_bf<Globals::hidden_dim> &
    get_rms_scale(megakernel::state<Config> &s) {
        return *reinterpret_cast<kittens::sv_bf<Globals::hidden_dim> *>(
            s.pages[get_activation_page(s)].ptr(sizeof(kittens::sv_bf<Globals::hidden_dim>)));
    }

    __device__ static inline uint8_t *get_output_start(megakernel::state<Config> &s, int stage) {
        return (uint8_t *)s.scratch() + (stage * SCRATCH_BYTES_PER_STAGE);
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
        if constexpr (traits::has_quant) {
            for (int i = 0; i < traits::scale_pipeline_stages; i++) {
                init_semaphore(scales_arrived(s, i), 1);
                init_semaphore(scales_finished(s, i), Config::NUM_CONSUMER_WARPS);
            }
        }
        init_semaphore(rms_scale_arrived(s), 1);
        return SEM_COUNT;
    }

    // Loader: RMS scale + quantization scale pre-loading + weights
    __device__ static inline void loader_loop(megakernel::state<Config> &s,
                                              const Globals &g, int layer_idx) {
        parsed_instruction inst{s};

        // Load RMS scale
        if (kittens::laneid() == 0) {
            int activation_page = get_activation_page(s);
            s.wait_page_ready(activation_page);
            auto &rms_scale = get_rms_scale(s);
            auto &sem = rms_scale_arrived(s);
            kittens::tma::expect(sem, rms_scale);
            kittens::tma::load_async<kittens::cache_policy::EVICT_LAST>(
                rms_scale, g.*RmsPtr, {layer_idx, 0}, sem);
        }

        // Weight loading with quantization scale pre-fetch
        if (kittens::laneid() == 0) {
            int input_stage = 0;
            for (int iter = 0; iter < inst.iters; iter++) {
                kittens::wait(weights_finished(s, input_stage),
                     (iter % (2 * INPUT_PIPELINE_STAGES)) < INPUT_PIPELINE_STAGES);

                if constexpr (traits::has_quant) {
                    int scale_stage = iter % traits::scale_pipeline_stages;
                    kittens::wait(scales_finished(s, scale_stage),
                         (iter % (2 * traits::scale_pipeline_stages)) <
                             traits::scale_pipeline_stages);
                    pipeline_specifics::load_scales_weight(s, g, inst, iter,
                        scales_arrived(s, scale_stage));
                }

                auto &sem = weights_arrived(s, input_stage);
                kittens::tma::expect_bytes(sem, sizeof(kittens::bf16) * Globals::hidden_dim * 16);

#pragma unroll
                for (int i = 0; i < STAGE_PAGES; i++) {
                    int weight_page = get_weight_page(s, input_stage, i);
                    if (iter < INPUT_PIPELINE_STAGES) {
                        s.wait_page_ready(weight_page);
                    }
                    auto &weight_chunk = reinterpret_cast<kittens::st_bf<16, 512> &>(
                        s.pages[weight_page]);
                    pipeline_specifics::load_iter(s, g, inst, iter, i,
                                                  weight_chunk, sem);
                }
                input_stage = (input_stage + 1) % INPUT_PIPELINE_STAGES;
            }
        } else {
            auto needed_pages =
                1 + min(inst.iters, INPUT_PIPELINE_STAGES) * STAGE_PAGES;
            if (kittens::laneid() >= needed_pages && kittens::laneid() < Config::NUM_PAGES) {
                auto pid = s.pid(kittens::laneid());
                s.wait_page_ready(pid);
                s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
            }
        }
    }

    // Consumer: dequantize + RMS norm + matvec
    __device__ static inline void consumer_loop(megakernel::state<Config> &s,
                                                 const Globals &g) {
        using sv_t = kittens::sv_bf<REDUCTION_DIM_PER_WARP>;
        auto &rms_scale_smem =
            reinterpret_cast<sv_t *>(&get_rms_scale(s))[kittens::warpid()];
        auto &activations_smem =
            reinterpret_cast<sv_t *>(&get_activations(s))[kittens::warpid()];

        if (kittens::laneid() == 0 && kittens::warpid() == 0) {
            s.record(megakernel::TEVENT_AT_GMEM_WAIT);
            pipeline_specifics::gmem_wait(g, s);
            s.record(megakernel::TEVENT_DONE_GMEM_WAIT);
        }
        kittens::group<Config::NUM_CONSUMER_WARPS>::sync(3);

        kittens::warp::load(activations_smem, g.*ActPtr, {kittens::warpid()});
        auto activation_page = get_activation_page(s);
        kittens::wait(rms_scale_arrived(s), 0);

        auto activations_vec = rms_norm<Config>(
            rms_scale_smem, activations_smem, g.rms_norm_eps,
            get_output_start(s, OUTPUT_PIPELINE_STAGES));
        kittens::warp::sync();
        s.warp_finish_page(activation_page, 1);

        // Matvec with optional dequantization
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
                kittens::st_bf<16, REDUCTION_DIM_PER_WARP> *>(
                s.pages[weight_page].ptr())[kittens::warpid() % WARPS_PER_PAGE];

            if constexpr (traits::has_quant) {
                int scale_stage = i % traits::scale_pipeline_stages;
                kittens::wait(scales_arrived(s, scale_stage),
                     (i % (2 * traits::scale_pipeline_stages)) >=
                         traits::scale_pipeline_stages);
                pipeline_specifics::dequant_weight(s, g, inst, i, weights_smem);
            }

            kittens::sv_fl<16> &out_smem = *reinterpret_cast<kittens::sv_fl<16> *>(
                get_output_start(s, output_stage) +
                (kittens::warpid() * SCRATCH_BYTES_PER_WARP));

            matvec(out_smem, weights_smem, activations_vec);

            kittens::warp::sync();
            kittens::warp::arrive(outputs_arrived(s, output_stage));
            kittens::warp::arrive(weights_finished(s, input_stage));

            if constexpr (traits::has_quant) {
                int scale_stage = i % traits::scale_pipeline_stages;
                kittens::warp::arrive(scales_finished(s, scale_stage));
            }

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

    // Storer (same as base)
    template <int iter_scale = 1>
    __device__ static inline void storer_loop(megakernel::state<Config> &s,
                                              const Globals &g) {
        parsed_instruction inst{s};
        int output_stage = 0;
        for (int i = 0; i < inst.iters; i++) {
            kittens::wait(outputs_arrived(s, output_stage),
                 (i % (2 * OUTPUT_PIPELINE_STAGES)) >= OUTPUT_PIPELINE_STAGES);
            pipeline_specifics::store(s, g, inst, i, output_stage);
            if ((i + 1) % iter_scale == 0) {
                for (int j = 0; j < iter_scale; j++) {
                    kittens::warp::arrive(outputs_finished(s, (i - j) % OUTPUT_PIPELINE_STAGES));
                }
            }
            output_stage = (output_stage + 1) % OUTPUT_PIPELINE_STAGES;
        }
    }
};
