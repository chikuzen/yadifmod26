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


#ifndef YADIF_MOD_H
#define YADIF_MOD_H

#include <cstdint>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <avisynth.h>

#define YADIF_MOD_26_VERSION "0.0.0"


typedef IScriptEnvironment ise_t;

typedef enum {
    NO_SIMD,
    USE_SSE2,
    USE_AVX2,
} arch_t;


typedef void(__stdcall *proc_filter_t)(
    const uint8_t* currp, const uint8_t* prevp, const uint8_t* nextp,
    const uint8_t* fm_prev, const uint8_t* fm_next, const uint8_t* edeintp,
    uint8_t* dstp, const int width, const int cpitch, const int ppitch,
    const int npitch, const int fm_ppitch, const int fm_npitch,
    const int epitch2, const int dpitch2, const int count);

proc_filter_t get_main_proc(bool sp_check, arch_t arcg);

class YadifMod : public GenericVideoFilter {
    PClip edeint;
    VideoInfo viSrc;
    int order;
    int field;
    int mode;
    int numPlanes;

    proc_filter_t mainProc;

public:
    YadifMod(PClip child, PClip edeint, int order, int field, int mode,
             arch_t arch);
    ~YadifMod() {}
    PVideoFrame __stdcall GetFrame(int n, ise_t* env);
};


#endif

