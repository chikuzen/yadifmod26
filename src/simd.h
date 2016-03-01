/*
**   Copyright (C) 2016 OKA Motofumi
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


#ifndef YADIF_MOD_SIMD_H
#define YADIF_MOD_SIMD_H

#include <immintrin.h>

#define SFINLINE static __forceinline


SFINLINE __m128i load(const __m128i* p)
{
    return _mm_loadu_si128(p);
}

SFINLINE __m256i load(const __m256i* p)
{
    return _mm256_loadu_si256(p);
}

SFINLINE void stream(__m128i* p, const __m128i& x)
{
    _mm_stream_si128(p, x);
}

SFINLINE void stream(__m256i* p, const __m256i& x)
{
    _mm256_stream_si256(p, x);
}

SFINLINE __m128i average(const __m128i& x, const __m128i& y)
{
    return _mm_avg_epu8(x, y);
}

SFINLINE __m256i average(const __m256i& x, const __m256i& y)
{
    return _mm256_avg_epu8(x, y);
}

SFINLINE __m128i max(const __m128i& x, const __m128i& y)
{
    return _mm_max_epu8(x, y);
}

SFINLINE __m128i max(const __m128i& x, const __m128i& y, const __m128i& z)
{
    return max(max(x, y), z);
}

SFINLINE __m256i max(const __m256i& x, const __m256i& y)
{
    return _mm256_max_epu8(x, y);
}

SFINLINE __m256i max(const __m256i& x, const __m256i& y, const __m256i& z)
{
    return max(max(x, y), z);
}

SFINLINE __m128i min(const __m128i& x, const __m128i& y)
{
    return _mm_min_epu8(x, y);
}

SFINLINE __m128i min(const __m128i& x, const __m128i& y, const __m128i& z)
{
    return min(min(x, y), z);
}

SFINLINE __m256i min(const __m256i& x, const __m256i& y)
{
    return _mm256_min_epu8(x, y);
}

SFINLINE __m256i min(const __m256i& x, const __m256i& y, const __m256i& z)
{
    return min(min(x, y), z);
}

SFINLINE __m128i abs_diff(const __m128i& x, const __m128i& y)
{
    return _mm_subs_epu8(max(x, y), min(x, y));
}

SFINLINE __m256i abs_diff(const __m256i& x, const __m256i& y)
{
    return _mm256_subs_epu8(max(x, y), min(x, y));
}

SFINLINE __m128i clamp(const __m128i& x, const __m128i& min_, const __m128i& max_)
{
    return max(min(x, max_), min_);
}

SFINLINE __m256i clamp(const __m256i& x, const __m256i& min_, const __m256i& max_)
{
    return max(min(x, max_), min_);
}

SFINLINE __m128i cvt8to16lo(const __m128i& x)
{
    const __m128i zero = _mm_setzero_si128();
    return _mm_unpacklo_epi8(x, zero);
}

SFINLINE __m256i cvt8to16lo(const __m256i& x)
{
    const __m256i zero = _mm256_setzero_si256();
    return _mm256_unpacklo_epi8(x, zero);
}

SFINLINE __m128i cvt8to16hi(const __m128i& x)
{
    const __m128i zero = _mm_setzero_si128();
    return _mm_unpackhi_epi8(x, zero);
}

SFINLINE __m256i cvt8to16hi(const __m256i& x)
{
    const __m256i zero = _mm256_setzero_si256();
    return _mm256_unpackhi_epi8(x, zero);
}

SFINLINE __m128i packus16(const __m128i& x, const __m128i& y)
{
    return _mm_packus_epi16(x, y);
}

SFINLINE __m256i packus16(const __m256i& x, const __m256i& y)
{
    return _mm256_packus_epi16(x, y);
}

SFINLINE __m128i div2(const __m128i& x)
{
    return  packus16(_mm_srli_epi16(cvt8to16lo(x), 1),
                     _mm_srli_epi16(cvt8to16hi(x), 1));
}

SFINLINE __m256i div2(const __m256i& x)
{
    return  packus16(_mm256_srli_epi16(cvt8to16lo(x), 1),
                     _mm256_srli_epi16(cvt8to16hi(x), 1));
}

SFINLINE __m128i sublo_epu8(const __m128i& x, const __m128i& y)
{
    return _mm_sub_epi16(cvt8to16lo(x), cvt8to16lo(y));
}

SFINLINE __m128i subhi_epu8(const __m128i& x, const __m128i& y)
{
    return _mm_sub_epi16(cvt8to16hi(x), cvt8to16hi(y));
}

SFINLINE __m256i sublo_epu8(const __m256i& x, const __m256i& y)
{
    return _mm256_sub_epi16(cvt8to16lo(x), cvt8to16lo(y));
}

SFINLINE __m256i subhi_epu8(const __m256i& x, const __m256i& y)
{
    return _mm256_sub_epi16(cvt8to16hi(x), cvt8to16hi(y));
}

SFINLINE __m128i max16(const __m128i& x, const __m128i& y)
{
    return _mm_max_epi16(x, y);
}

SFINLINE __m128i max16(const __m128i& x, const __m128i& y, const __m128i& z)
{
    return max16(max16(x, y), z);
}

SFINLINE __m256i max16(const __m256i& x, const __m256i& y)
{
    return _mm256_max_epi16(x, y);
}

SFINLINE __m256i max16(const __m256i& x, const __m256i& y, const __m256i& z)
{
    return max16(max16(x, y), z);
}

SFINLINE __m128i min16(const __m128i& x, const __m128i& y)
{
    return _mm_min_epi16(x, y);
}

SFINLINE __m128i min16(const __m128i& x, const __m128i& y, const __m128i& z)
{
    return min16(min16(x, y), z);
}

SFINLINE __m256i min16(const __m256i& x, const __m256i& y)
{
    return _mm256_min_epi16(x, y);
}

SFINLINE __m256i min16(const __m256i& x, const __m256i& y, const __m256i& z)
{
    return min16(min16(x, y), z);
}

SFINLINE __m128i invert_sign(const __m128i& x)
{
    const __m128i one = _mm_set1_epi16(0x0001);
    const __m128i all = _mm_cmpeq_epi16(one, one);
    return _mm_add_epi16(_mm_xor_si128(x, all), one);
}

SFINLINE __m256i invert_sign(const __m256i& x)
{
    const __m256i one = _mm256_set1_epi16(0x0001);
    const __m256i all = _mm256_cmpeq_epi16(one, one);
    return _mm256_add_epi16(_mm256_xor_si256(x, all), one);
}

SFINLINE __m128i sub(const __m128i& x, const __m128i& y)
{
    return _mm_subs_epu8(x, y);
}

SFINLINE __m256i sub(const __m256i& x, const __m256i& y)
{
    return _mm256_subs_epu8(x, y);
}

SFINLINE __m128i add(const __m128i& x, const __m128i& y)
{
    return _mm_adds_epu8(x, y);
}

SFINLINE __m256i add(const __m256i& x, const __m256i& y)
{
    return _mm256_adds_epu8(x, y);
}
#endif

