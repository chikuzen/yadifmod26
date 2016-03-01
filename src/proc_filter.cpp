/*
**   Rewite for Avisynth 2.6 and Avisynth+
**   Copyright (C) 2016 OKA Motofumi
**   Modification of Fizick's yadif avisynth filter.
**   Copyright (C) 2007 Kevin Stone
**   Yadif C-plugin for Avisynth 2.5 - Yet Another DeInterlacing Filter
**   Copyright (C) 2007 Alexander G. Balakhnin aka Fizick  http://avisynth.org.ru
**   Port of YADIF filter from MPlayer
**   Copyright (C) 2006 Michael Niedermayer <michaelni@gmx.at>
**
**   This program is free software; you can redistribute it and/or modify
**   it under the terms of the GNU General Public License as published by
**   the Free Software Foundation.
**
**   This program is distributed in the hope that it will be useful,
**   but WITHOUT ANY WARRANTY; without even the implied warranty of
**   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**   GNU General Public License for more details.
**
**   You should have received a copy of the GNU General Public License
**   along with this program; if not, write to the Free Software
**   Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
*/

#include <cstdint>
#include <algorithm>
#include "yadifmod.h"
#include "simd.h"


static inline int average(int x, int y)
{
    return (x + y + 1) / 2;
}

static inline uint8_t clamp(int val, int min, int max)
{
    return (uint8_t)std::max(std::min(val, max), min);
}


template <bool SP_CHECK>
static void __stdcall
proc_filter_c(const uint8_t* curr_pre, const uint8_t* curr_nxt,
              const uint8_t* prev_pre, const uint8_t* prev_nxt,
              const uint8_t* next_pre, const uint8_t* next_nxt,
              const uint8_t* fm_prev_top, const uint8_t* fm_prev_mdl,
              const uint8_t* fm_prev_btm, const uint8_t* fm_next_top,
              const uint8_t* fm_next_mdl, const uint8_t* fm_next_btm,
              const uint8_t* edeintp, uint8_t* dstp,
              const int width, const int height,
              const int cpitch2, const int ppitch2, const int npitch2,
              const int fm_ppitch, const int fm_npitch, const int epitch2,
              const int dpitch2, const int begin, const int end)
{
    using std::abs;
    using std::min;
    using std::max;

    for (int y = begin; y <= end; y += 2) {
        for (int x = 0; x < width; ++x) {
            const int p1 = curr_pre[x];
            const int p2 = average(fm_prev_mdl[x], fm_next_mdl[x]);
            const int p3 = curr_nxt[x];
            const int d0 = abs(fm_prev_mdl[x] - fm_next_mdl[x]) / 2;
            const int d1 = average(abs(prev_pre[x] - p1), abs(prev_nxt[x] - p3));
            const int d2 = average(abs(next_pre[x] - p1), abs(next_nxt[x] - p3));
            int diff = max({ d0, d1, d2 });

            if (SP_CHECK) {
                const int p1_ = p2 - p1;
                const int p3_ = p2 - p3;
                const int p0 = average(fm_prev_top[x], fm_next_top[x]) - p1;
                const int p4 = average(fm_prev_btm[x], fm_next_btm[x]) - p3;
                const int maxs = max({ p1_, p3_, min(p0, p4) });
                const int mins = min({ p1_, p3_, max(p0, p4) });
                diff = max({ diff, mins, -maxs });
            }

            dstp[x] = clamp(edeintp[x], p2 - diff, p2 + diff);
        }
        curr_pre += cpitch2;
        curr_nxt += cpitch2;
        prev_pre += ppitch2;
        prev_nxt += ppitch2;
        next_pre += npitch2;
        next_nxt += npitch2;
        fm_prev_top += fm_ppitch;
        fm_prev_mdl += fm_ppitch;
        fm_prev_btm += fm_ppitch;
        fm_next_top += fm_npitch;
        fm_next_mdl += fm_npitch;
        fm_next_btm += fm_npitch;
        dstp += dpitch2;
        edeintp += epitch2;
    }
}


template <bool SP_CHECK, typename T>
static void __stdcall
proc_filter(const uint8_t* curr_pre, const uint8_t* curr_nxt,
    const uint8_t* prev_pre, const uint8_t* prev_nxt,
    const uint8_t* next_pre, const uint8_t* next_nxt,
    const uint8_t* fm_prev_top, const uint8_t* fm_prev_mdl,
    const uint8_t* fm_prev_btm, const uint8_t* fm_next_top,
    const uint8_t* fm_next_mdl, const uint8_t* fm_next_btm,
    const uint8_t* edeintp, uint8_t* dstp,
    const int width, const int height,
    const int cpitch2, const int ppitch2, const int npitch2,
    const int fm_ppitch, const int fm_npitch, const int epitch2,
    const int dpitch2, const int begin, const int end)
{
    for (int y = begin; y <= end; y += 2) {
        for (int x = 0; x < width; x += sizeof(T)) {
            const T fpm = load((T*)(fm_prev_mdl + x));
            const T fnm = load((T*)(fm_next_mdl + x));

            const T p1 = load((T*)(curr_pre + x));
            const T p2 = average(fpm, fnm);
            const T p3 = load((T*)(curr_nxt + x));

            const T d0 = div2(abs_diff(fpm, fnm));
            const T d1 = average(abs_diff(load((T*)(prev_pre + x)), p1),
                                 abs_diff(load((T*)(prev_nxt + x)), p3));
            const T d2 = average(abs_diff(load((T*)(next_pre + x)), p1),
                                 abs_diff(load((T*)(next_nxt + x)), p3));

            T diff = max(d0, d1, d2);

            if (SP_CHECK) {
                const T avg0 = average(load((T*)(fm_prev_top + x)),
                                       load((T*)(fm_next_top + x)));
                const T p0_lo = sublo_epu8(avg0, p1);
                const T p0_hi = subhi_epu8(avg0, p1);

                const T avg1 = average(load((T*)(fm_prev_btm + x)),
                                       load((T*)(fm_next_btm + x)));
                const T p4_lo = sublo_epu8(avg1, p3);
                const T p4_hi = subhi_epu8(avg1, p3);

                const T p1_lo = sublo_epu8(p2, p1);
                const T p1_hi = subhi_epu8(p2, p1);

                const T p3_lo = sublo_epu8(p2, p3);
                const T p3_hi = subhi_epu8(p2, p3);

                const T maxs_lo = max16(p1_lo, p3_lo, min16(p0_lo, p4_lo));
                const T maxs_hi = max16(p1_hi, p3_hi, min16(p0_hi, p4_hi));

                const T mins_lo = min16(p1_lo, p3_lo, max16(p0_lo, p4_lo));
                const T mins_hi = min16(p1_hi, p3_hi, max16(p0_hi, p4_hi));

                const T diff_lo = max16(cvt8to16lo(diff), mins_lo, invert_sign(maxs_lo));
                const T diff_hi = max16(cvt8to16hi(diff), mins_hi, invert_sign(maxs_hi));

                diff = packus16(diff_lo, diff_hi);
            }

            const T dst = clamp(load((T*)(edeintp + x)), sub(p2, diff), add(p2, diff));
            stream((T*)(dstp + x), dst);
        }
        curr_pre += cpitch2;
        curr_nxt += cpitch2;
        prev_pre += ppitch2;
        prev_nxt += ppitch2;
        next_pre += npitch2;
        next_nxt += npitch2;
        fm_prev_top += fm_ppitch;
        fm_prev_mdl += fm_ppitch;
        fm_prev_btm += fm_ppitch;
        fm_next_top += fm_npitch;
        fm_next_mdl += fm_npitch;
        fm_next_btm += fm_npitch;
        dstp += dpitch2;
        edeintp += epitch2;
    }
}


proc_filter_t get_main_proc(bool sp_check, arch_t arch)
{
    if (arch == NO_SIMD) {
        return sp_check ? proc_filter_c<true> : proc_filter_c<false>;
    }
    if (arch == USE_SSE2) {
        return sp_check ? proc_filter<true, __m128i> : proc_filter<false, __m128i>;
    }
    return sp_check ? proc_filter<true, __m256i> : proc_filter<false, __m256i>;
}
