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
    return static_cast<uint8_t>(std::max(std::min(val, max), min));
}


template <bool SP_CHECK>
static void __stdcall
proc_filter_c(const uint8_t* currp, const uint8_t* prevp, const uint8_t* nextp,
              const uint8_t* fmprev, const uint8_t* fmnext,
              const uint8_t* edeintp, uint8_t* dstp, const int width,
              const int cpitch, const int ppitch, const int npitch,
              const int fmppitch, const int fmnpitch, const int epitch2,
              const int dpitch2, const int count)
{
    using std::abs;
    using std::min;
    using std::max;

    const uint8_t* ct = currp - cpitch;
    const uint8_t* cb = currp + cpitch;
    const uint8_t* pt = prevp - ppitch;
    const uint8_t* pb = prevp + ppitch;
    const uint8_t* nt = nextp - npitch;
    const uint8_t* nb = nextp + npitch;
    const uint8_t* fmpt = fmprev - fmppitch;
    const uint8_t* fmpb = fmprev + fmppitch;
    const uint8_t* fmnt = fmnext - fmnpitch;
    const uint8_t* fmnb = fmnext + fmnpitch;

    for (int y = 0; y < count; ++y) {
        for (int x = 0; x < width; ++x) {
            const int p1 = ct[x];
            const int p2 = average(fmprev[x], fmnext[x]);
            const int p3 = cb[x];
            const int d0 = abs(fmprev[x] - fmnext[x]) / 2;
            const int d1 = average(abs(pt[x] - p1), abs(pb[x] - p3));
            const int d2 = average(abs(nt[x] - p1), abs(nb[x] - p3));
            int diff = max({ d0, d1, d2 });

            if (SP_CHECK) {
                const int p1_ = p2 - p1;
                const int p3_ = p2 - p3;
                const int p0 = average(fmpt[x], fmnt[x]) - p1;
                const int p4 = average(fmpb[x], fmnb[x]) - p3;
                const int maxs = max({ p1_, p3_, min(p0, p4) });
                const int mins = min({ p1_, p3_, max(p0, p4) });
                diff = max({ diff, mins, -maxs });
            }

            dstp[x] = clamp(edeintp[x], p2 - diff, p2 + diff);
        }

        ct += cpitch * 2;
        cb += cpitch * 2;
        pt += ppitch * 2;
        pb += ppitch * 2;
        nt += npitch * 2;
        nb += npitch * 2;
        fmprev += fmppitch;
        fmpt += fmppitch;
        fmpb += fmppitch;
        fmnext += fmnpitch;
        fmnt += fmnpitch;
        fmnb += fmnpitch;
        dstp += dpitch2;
        edeintp += epitch2;
    }
}


template <bool SP_CHECK, typename T>
static void __stdcall
proc_filter(const uint8_t* currp, const uint8_t* prevp, const uint8_t* nextp,
            const uint8_t* fmprev, const uint8_t* fmnext,
            const uint8_t* edeintp, uint8_t* dstp, const int width,
            const int cpitch, const int ppitch, const int npitch,
            const int fmppitch, const int fmnpitch, const int epitch2,
            const int dpitch2, const int count)
{
    const uint8_t* ct = currp - cpitch;
    const uint8_t* cb = currp + cpitch;
    const uint8_t* pt = prevp - ppitch;
    const uint8_t* pb = prevp + ppitch;
    const uint8_t* nt = nextp - npitch;
    const uint8_t* nb = nextp + npitch;
    const uint8_t* fmpt = fmprev - fmppitch;
    const uint8_t* fmpb = fmprev + fmppitch;
    const uint8_t* fmnt = fmnext - fmnpitch;
    const uint8_t* fmnb = fmnext + fmnpitch;

    for (int y = 0; y < count; ++y) {
        for (int x = 0; x < width; x += sizeof(T)) {
            const T fpm = load(reinterpret_cast<const T*>(fmprev + x));
            const T fnm = load(reinterpret_cast<const T*>(fmnext + x));

            const T p1 = load(reinterpret_cast<const T*>(ct + x));
            const T p2 = average(fpm, fnm);
            const T p3 = load(reinterpret_cast<const T*>(cb + x));

            const T d0 = div2(abs_diff(fpm, fnm));
            const T d1 = average(abs_diff(load(reinterpret_cast<const T*>(pt + x)), p1),
                                 abs_diff(load(reinterpret_cast<const T*>(pb + x)), p3));
            const T d2 = average(abs_diff(load(reinterpret_cast<const T*>(nt + x)), p1),
                                 abs_diff(load(reinterpret_cast<const T*>(nb + x)), p3));

            T diff = max(d0, d1, d2);

            if (SP_CHECK) {
                const T avg0 = average(load(reinterpret_cast<const T*>(fmpt + x)),
                                       load(reinterpret_cast<const T*>(fmnt + x)));
                const T p0_lo = sublo_epu8(avg0, p1);
                const T p0_hi = subhi_epu8(avg0, p1);

                const T avg1 = average(load(reinterpret_cast<const T*>(fmpb + x)),
                                       load(reinterpret_cast<const T*>(fmnb + x)));
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

            T spatial_pred = load(reinterpret_cast<const T*>(edeintp + x));
            const T dst = clamp(spatial_pred, sub(p2, diff), add(p2, diff));
            stream(reinterpret_cast<T*>(dstp + x), dst);
        }
        ct += cpitch * 2;
        cb += cpitch * 2;
        pt += ppitch * 2;
        pb += ppitch * 2;
        nt += npitch * 2;
        nb += npitch * 2;
        fmpt += fmppitch;
        fmprev += fmppitch;
        fmpb += fmppitch;
        fmnt += fmnpitch;
        fmnext += fmnpitch;
        fmnb += fmnpitch;
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
