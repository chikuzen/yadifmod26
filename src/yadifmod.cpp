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


extern proc_filter_t get_main_proc(bool sp_check, arch_t arch);


YadifMod::YadifMod(PClip c, PClip e, int o, int f, int m, arch_t arch) :
    GenericVideoFilter(c), edeint(e), order(o), field(f), mode(m)
{
    numPlanes = vi.IsY8() ? 1 : 3;

    memcpy(&viSrc, &vi, sizeof(vi));

    if (mode == 1 || mode == 3) {
        vi.num_frames *= 2;
        vi.SetFPS(vi.fps_numerator * 2, vi.fps_denominator);
    }

    if (order == -1) {
        order = child->GetParity(0) ? 1 : 0;
    }
    if (field == -1) {
        field = order;
    }

    main_proc = get_main_proc(mode < 2, arch);
}


static inline int mapn(int n, const int& nf)
{
    return std::min(std::max(n, 0), nf - 1);
}


PVideoFrame __stdcall YadifMod::GetFrame(int n, ise_t* env)
{
    const int planes[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
    const int nf = viSrc.num_frames;

    auto edeint = this->edeint->GetFrame(n, env);

    int fieldt = field;
    if (mode == 1 || mode == 3) {
        fieldt = (n & 1) ? 1 - order : order;
        n /= 2;
    }

    auto curr = child->GetFrame(n, env);
    auto prev = child->GetFrame(std::max(n - 1, 0), env);
    auto next = child->GetFrame(std::min(n + 1, nf - 1), env);
    auto dst = env->NewVideoFrame(vi, 32);

    for (int p = 0; p < numPlanes; ++p) {

        const int plane = planes[p];

        const uint8_t* currp = curr->GetReadPtr(plane);
        const uint8_t* prevp = prev->GetReadPtr(plane);
        const uint8_t* nextp = next->GetReadPtr(plane);

        const int width = curr->GetRowSize(plane);
        const int height = curr->GetHeight(plane);
        const int cpitch = curr->GetPitch(plane);
        const int ppitch = prev->GetPitch(plane);
        const int npitch = next->GetPitch(plane);

        const int begin = 2 + fieldt;
        const int end = height - 4 + fieldt;

        const uint8_t* curr_pre = currp + cpitch * (begin - 1);
        const uint8_t* curr_nxt = curr_pre + 2 * cpitch;
        const uint8_t* prev_pre = prevp + ppitch * (begin - 1);
        const uint8_t* prev_nxt = prev_pre + 2 * ppitch;
        const uint8_t* next_pre = nextp + npitch * (begin - 1);
        const uint8_t* next_nxt = next_pre + 2 * npitch;

        const uint8_t *fm_prev_top, *fm_prev_mdl, *fm_prev_btm,
                      *fm_next_top, *fm_next_mdl, *fm_next_btm;
        int fm_ppitch, fm_npitch;
        if (fieldt != order) {
            fm_ppitch = cpitch * 2;
            fm_npitch = npitch * 2;
            fm_prev_mdl = currp + begin * cpitch;
            fm_next_mdl = nextp + begin * npitch;
        } else {
            fm_ppitch = ppitch * 2;
            fm_npitch = cpitch * 2;
            fm_prev_mdl = prevp + begin * ppitch;
            fm_next_mdl = currp + begin * cpitch;
        }
        fm_prev_top = fm_prev_mdl - fm_ppitch;
        fm_prev_btm = fm_prev_mdl + fm_ppitch;
        fm_next_top = fm_next_mdl - fm_npitch;
        fm_next_btm = fm_next_mdl + fm_npitch;

        const uint8_t* edeintp = edeint->GetReadPtr(plane);
        uint8_t* dstp = dst->GetWritePtr(plane);

        const int epitch = edeint->GetPitch(plane);
        const int dpitch = dst->GetPitch(plane);

        if (fieldt == 0) {
            memcpy(dstp, currp + cpitch, width);
            memcpy(dstp + dpitch * (height - 2),
                   edeintp + epitch * (height - 2), width);
        } else {
            memcpy(dstp + dpitch, edeintp + epitch, width);
            memcpy(dstp + dpitch * (height - 1),
                   currp + cpitch * (height - 2), width);
        }
        env->BitBlt(dstp + (1 - fieldt) * dpitch, 2 * dpitch,
                    currp + (1 - fieldt) * cpitch, 2 * cpitch, width, height / 2);

        main_proc(curr_pre, curr_nxt, prev_pre, prev_nxt, next_pre, next_nxt,
                  fm_prev_top, fm_prev_mdl, fm_prev_btm,
                  fm_next_top, fm_next_mdl, fm_next_btm,
                  edeintp + begin * epitch, dstp + begin * dpitch,
                  width, height, 2 * cpitch, 2 * ppitch, 2 * npitch,
                  fm_ppitch, fm_npitch, 2 * epitch, 2 * dpitch, begin, end);
    }

    return dst;
}


extern int has_sse2(void);
extern int has_avx2(void);

static arch_t get_arch(int opt)
{
    if (opt == 0 || !has_sse2()) {
        return NO_SIMD;
    }
    if (opt == 1 || !has_avx2()) {
        return USE_SSE2;
    }
    return USE_AVX2;
}

static void validate(bool cond, const char* msg, ise_t* env)
{
    if (!cond) {
        env->ThrowError("yadifmod: %s", msg);
    }
}

static AVSValue __cdecl
create_yadifmod(AVSValue args, void* user_data, ise_t* env)
{
    validate(args[4].Defined(), "an edeint clip must be specified.", env);

    PClip child = args[0].AsClip();
    const VideoInfo& vi = child->GetVideoInfo();
    validate(vi.IsPlanar(), "input clip must be a planar format.", env);

    PClip edeint = args[4].AsClip();
    const VideoInfo& vi_ed = edeint->GetVideoInfo();
    validate(vi.IsSameColorspace(vi_ed),
             "edeint clip's colorspace doesn't match.", env);
    validate(vi.width == vi_ed.width && vi.height == vi_ed.height,
             "input and edeint must be the same resolution.", env);

    int order = args[1].AsInt(-1);
    validate(order >= -1 && order <= 1,
             "order must be set to -1, 0 or 1.", env);

    int field = args[2].AsInt(-1);
    validate(field >= -1 && field <= 1,
             "field must be set to -1, 0 or 1.", env);

    int mode = args[3].AsInt(0);
    validate(mode >= 0 && mode <= 3,
             "mode must be set to 0, 1, 2 or 3", env);
    validate(vi.num_frames * ((mode & 1) ? 2 : 1) == vi_ed.num_frames,
             "edeint clip's number of frames doesn't match.", env);

    int opt = args[5].AsInt(1);
    validate(opt >= -1 && opt <= 2,
             "opt must be set to -1, 0, 1 or 2.", env);
    arch_t arch = get_arch(opt);

    return new YadifMod(child, edeint, order, field, mode, arch);
}


static const AVS_Linkage* AVS_linkage = nullptr;


extern "C" __declspec(dllexport) const char* __stdcall
AvisynthPluginInit3(ise_t* env, const AVS_Linkage* vectors)
{
    AVS_linkage = vectors;
    env->AddFunction("yadifmod",
                     "c"
                     "[order]i"
                     "[field]i"
                     "[mode]i"
                     "[edeint]c"
                     "[opt]i",
                     create_yadifmod, nullptr);
    return "yadifmod for avs2.6 ver. " YADIF_MOD_26_VERSION " by OKA Motofumi";
}
