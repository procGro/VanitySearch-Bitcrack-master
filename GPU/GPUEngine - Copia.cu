/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef WIN64
#include <unistd.h>
#include <stdio.h>
#endif

#include "GPUEngine.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "../hash/sha256.h"
#include "../hash/ripemd160.h"
#include "../Timer.h"

#include "GPUGroup.h"
#include "GPUMath.h"
#include "GPUHash.h"
#include "GPUBase58.h"
#include "GPUWildcard.h"
#include "GPUCompute.h"
#include <iostream>

#include <omp.h>

//uint64_t _2Gnx[4] = { 0x0646E23FD5F51508ULL,0xD8C39CABD5AC1CA1ULL,0xEA2A6E3E172DE238ULL,0x8282263212C609D9ULL };  //256
//    //uint64_t _2Gny[4] = { 0xD31B6EAFF6E26CAFULL,0x62D613AC2F7B17BEULL,0x5E8256E830B60ACEULL,0x11F8A8098557DFE4ULL };
//
//    uint64_t _2Gnx[4] = { 0xD5B901B2E285131FULL,0xAAEC6ECDC813B088ULL,0xD664A18F66AD6240ULL,0x241FEBB8E23CBD77ULL }; //1024
//    uint64_t _2Gny[4] = { 0xABB3E66F2750026DULL,0xCD50FD0FBD0CB5AFULL,0xD6C420BD13981DF8ULL,0x513378D9FF94F8D3ULL };
//
//    //uint64_t _2Gnx[4] = { 0xEDCB63069B920471ULL, 0xFC318B85F423DE0DULL, 0xFCE4CC2983D8F8D9ULL, 0x5D1BDB4EA172FA79ULL }; //2048
//    //uint64_t _2Gny[4] = { 0x70330666F7B83103ULL, 0x79EB1E9996C56E7BULL, 0x794BB99438A22656ULL, 0x2843826779379E2EULL };

//__global__ void comp_keys_256(prefix_t* sPrefix, uint32_t* lookup32, uint64_t* keys, uint32_t* out) {
//
//    int xPtr = (blockIdx.x * blockDim.x) * 8;
//    int yPtr = xPtr + 4 * blockDim.x;
//
//    uint64_t* startx = keys + xPtr;
//    uint64_t* starty = keys + yPtr;
//
//    uint64_t dx[256 / 2 + 1][4];
//    uint64_t px[4];
//    uint64_t py[4];
//    uint64_t pyn[4];
//    uint64_t sx[4];
//    uint64_t sy[4];
//    uint64_t dy[4];
//    uint64_t _s[4];
//    uint64_t _p2[4];
//    uint8_t odd_py;
//    uint32_t h[5];
//
//    uint64_t _2Gnx[4] = { 0x0646E23FD5F51508ULL,0xD8C39CABD5AC1CA1ULL,0xEA2A6E3E172DE238ULL,0x8282263212C609D9ULL };  //256
//    uint64_t _2Gny[4] = { 0xD31B6EAFF6E26CAFULL,0x62D613AC2F7B17BEULL,0x5E8256E830B60ACEULL,0x11F8A8098557DFE4ULL };
//
//    //uint64_t _2Gnx[4] = { 0xD5B901B2E285131FULL,0xAAEC6ECDC813B088ULL,0xD664A18F66AD6240ULL,0x241FEBB8E23CBD77ULL }; //1024
//    //uint64_t _2Gny[4] = { 0xABB3E66F2750026DULL,0xCD50FD0FBD0CB5AFULL,0xD6C420BD13981DF8ULL,0x513378D9FF94F8D3ULL };
//
//    //uint64_t _2Gnx[4] = { 0xEDCB63069B920471ULL, 0xFC318B85F423DE0DULL, 0xFCE4CC2983D8F8D9ULL, 0x5D1BDB4EA172FA79ULL }; //2048
//    //uint64_t _2Gny[4] = { 0x70330666F7B83103ULL, 0x79EB1E9996C56E7BULL, 0x794BB99438A22656ULL, 0x2843826779379E2EULL };
//
//    //uint64_t _2Gnx[4] = { 0x73FCE5B551E5B739ULL, 0xE0B93833FD2222EDULL, 0x72F99CC6C6FC846DULL, 0x175E159F728B865AULL }; //4096
//    //uint64_t _2Gny[4] = { 0x6EFA6FFEE9FED695ULL, 0xACB5955ADD24345CULL, 0xA4EF97A51FF71F5EULL, 0xD3506E0D9E3C79EBULL };
//
//    //uint64_t _2Gnx[4] = { 0xCE78049E46BC47D6ULL, 0x30FDFEB5C6DA121BULL, 0xA5FFBCC8E139C621ULL, 0x423A013F03FF32D7ULL }; //8192
//    //uint64_t _2Gny[4] = { 0xD1236E6D8B548A34ULL, 0x720D8EC3524F009EULL, 0xA1179F7BBAF6B3C7ULL, 0xB91AE00FE1E1D970ULL }; 
//
//    // Load starting key
//    __syncthreads();
//    Load256A(sx, startx);
//    Load256A(sy, starty);
//    Load256(px, sx);
//    //Load256(py, sy);
//    /*Load256(px, sx);
//    Load256(py, sy);*/
//
//    uint32_t i;
//    uint32_t j;
//
//    for (j = 0; j < STEP_SIZE / 256; j++) {
//
//        // Check starting point
//        odd_py = sy[0] & 1;
//        //CHECK_PREFIX(256 / 2);
//
//        _GetHash160Comp(px, odd_py, (uint8_t*)h);
//        CheckPoint(h, j * 256 + 256 / 2, sPrefix, lookup32, out);
//
//        // Fill group with delta x
//
//        for (i = 0; i < 127; i++)
//            ModSub256(dx[i], Gx[i], sx);
//        ModSub256(dx[i], Gx[i], sx);  // For the first point
//        ModSub256(dx[i + 1], _2Gnx, sx);  // For the next center point
//
//        // Compute modular inverse
//        _ModInvGrouped256(dx);
//
//        ModNeg256(pyn, sy);
//
//        for (i = 0; i < 127; i++) {
//
//            __syncthreads();
//            // P = StartPoint + i*G
//            /*Load256(px, sx);
//            Load256(py, sy);*/
//
//            ModSub256(dy, Gy[i], sy);
//
//            _ModMult(_s, dy, dx[i]);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
//            _ModSqr(_p2, _s);             // _p2 = pow2(s)
//
//            ModSub256(px, _p2, sx);
//            ModSub256(px, Gx[i]);         // px = pow2(s) - p1.x - p2.x;
//
//            ModSub256(py, Gx[i], px);
//            _ModMult(py, _s);
//            ModSub256isOdd(py, Gy[i], &odd_py);
//
//            //ModSub256(py, Gy[i]);         // py = - p2.y - s*(ret.x-p2.x);
//            //odd_py = odd_py = py[0] & 1;
//
//
//            //CHECK_PREFIX(256 / 2 + (i + 1));
//
//            _GetHash160Comp(px, odd_py, (uint8_t*)h);
//            CheckPoint(h, j * 256 + 256 / 2 + (i + 1), sPrefix, lookup32, out);
//
//            __syncthreads();
//
//            // P = StartPoint - i*G, if (x,y) = i*G then (x,-y) = -i*G
//            //Load256(px, sx);
//            ModSub256(dy, pyn, Gy[i]);
//
//            _ModMult(_s, dy, dx[i]);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
//            _ModSqr(_p2, _s);             // _p = pow2(s)
//
//            ModSub256(px, _p2, sx);
//            ModSub256(px, Gx[i]);         // px = pow2(s) - p1.x - p2.x;
//
//            ModSub256(py, px, Gx[i]);
//            _ModMult(py, _s);             // py = s*(ret.x-p2.x)
//            ModSub256isOdd(py, Gy[i], py, &odd_py);// py = s*(ret.x-p2.x)
//
//
//            //ModSub256(py, Gy[i], py);     // py = - p2.y - s*(ret.x-p2.x);
//            //odd_py = py[0] & 1;
//
//
//
//            //CHECK_PREFIX(256 / 2 - (i + 1));
//
//            _GetHash160Comp(px, odd_py, (uint8_t*)h);
//            CheckPoint(h, j * 256 + 256 / 2 - (i + 1), sPrefix, lookup32, out);
//
//        }
//
//        // First point (startP - (GRP_SZIE/2)*G)
//        __syncthreads();
//        //Load256(px, sx);
//        ModNeg256(dy, Gy[i]);
//        ModSub256(dy, sy);
//
//
//        _ModMult(_s, dy, dx[i]);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
//        _ModSqr(_p2, _s);              // _p = pow2(s)
//
//        ModSub256(px, _p2, sx);
//        ModSub256(px, Gx[i]);         // px = pow2(s) - p1.x - p2.x;
//
//        ModSub256(py, px, Gx[i]);
//        _ModMult(py, _s);
//        ModSub256isOdd(py, Gy[i], py, &odd_py);// py = s*(ret.x-p2.x)
//
//        //ModSub256(py, Gy[i], py);     // py = - p2.y - s*(ret.x-p2.x);
//        //odd_py = py[0] & 1;
//
//        //CHECK_PREFIX(0);
//
//        _GetHash160Comp(px, odd_py, (uint8_t*)h);
//        CheckPoint(h, j * 256, sPrefix, lookup32, out);
//
//        i++;
//
//        // Next start point (startP + 256*G)
//        __syncthreads();
//        /*Load256(px, sx);
//        Load256(py, sy);*/
//        ModSub256(dy, _2Gny, sy);
//
//        _ModMult(_s, dy, dx[i]);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
//        _ModSqr(_p2, _s);             // _p2 = pow2(s)
//
//        ModSub256(px, _p2, sx);
//        ModSub256(px, _2Gnx);         // px = pow2(s) - p1.x - p2.x;
//
//        ModSub256(py, _2Gnx, px);
//        _ModMult(py, _s);             // py = - s*(ret.x-p2.x)
//        ModSub256(py, _2Gny);         // py = - p2.y - s*(ret.x-p2.x);
//
//
//        Load256(sx, px);
//        Load256(sy, py);
//
//    }
//
//    // Update starting point
//    __syncthreads();
//    Store256A(startx, px);
//    Store256A(starty, py);
//
//}

//__global__ void comp_keys_1024(prefix_t* sPrefix, uint32_t* lookup32, uint64_t* keys, uint32_t* out) {
//
//    uint64_t* startx = keys + (blockIdx.x * blockDim.x) * 8;
//    uint64_t* starty = keys + (blockIdx.x * blockDim.x) * 8 + 4 * blockDim.x;
//
//    uint64_t dx[1024 / 2 + 1][4];
//    uint64_t px[4];
//    uint64_t py[4];
//    uint64_t sxn[4];
//    uint64_t syn[4];
//    uint64_t sx_gx[4];
//    uint64_t sx[4];
//    uint64_t sy[4];
//    uint64_t dy[4];
//    uint64_t _s[4];
//    uint64_t _p2[4];
//    uint8_t odd_py;
//    uint32_t h[5];
//
//    //uint64_t _2Gnx[4] = { 0x0646E23FD5F51508ULL,0xD8C39CABD5AC1CA1ULL,0xEA2A6E3E172DE238ULL,0x8282263212C609D9ULL };  //256
//    //uint64_t _2Gny[4] = { 0xD31B6EAFF6E26CAFULL,0x62D613AC2F7B17BEULL,0x5E8256E830B60ACEULL,0x11F8A8098557DFE4ULL };
//
//    uint64_t _2Gnx[4] = { 0xD5B901B2E285131FULL,0xAAEC6ECDC813B088ULL,0xD664A18F66AD6240ULL,0x241FEBB8E23CBD77ULL }; //1024
//    uint64_t _2Gny[4] = { 0xABB3E66F2750026DULL,0xCD50FD0FBD0CB5AFULL,0xD6C420BD13981DF8ULL,0x513378D9FF94F8D3ULL };
//
//    //uint64_t _2Gnx[4] = { 0xEDCB63069B920471ULL, 0xFC318B85F423DE0DULL, 0xFCE4CC2983D8F8D9ULL, 0x5D1BDB4EA172FA79ULL }; //2048
//    //uint64_t _2Gny[4] = { 0x70330666F7B83103ULL, 0x79EB1E9996C56E7BULL, 0x794BB99438A22656ULL, 0x2843826779379E2EULL };
//
//
//    // Load starting key
//    __syncthreads();
//    Load256A(sx, startx);
//    Load256A(sy, starty);
//    Load256(px, sx);
//    //Load256(py, sy);
//    /*Load256(px, sx);
//    Load256(py, sy);*/
//
//    uint32_t i;
//    uint32_t j;
//
//    for (j = 0; j < STEP_SIZE / 1024; j++) {
//
//        // Check starting point
//        odd_py = sy[0] & 1;
//        //CHECK_PREFIX(2048 / 2);
//
//        _GetHash160Comp(px, odd_py, (uint8_t*)h);
//        CheckPoint(h, j * 1024 + 1024 / 2, sPrefix, lookup32, out);
//
//        // Fill group with delta x
//
//        for (i = 0; i < 1024/2; i++)
//            ModSub256(dx[i], Gx[i], sx);
//        ModSub256(dx[i], _2Gnx, sx);  // For the next center point
//
//        // Compute modular inverse
//        _ModInvGrouped1024(dx);
//
//        ModNeg256(syn, sy);
//        ModNeg256(sxn, sx);
//
//        for (i = 0; i < 1024/2-1; i++) {
//
//            __syncthreads();
//            // P = StartPoint + i*G
//            /*Load256(px, sx);
//            Load256(py, sy);*/
//
//            ModSub256(dy, Gy[i], sy);
//
//            _ModMult(_s, dy, dx[i]);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
//            _ModSqr(_p2, _s);             // _p2 = pow2(s)
//
//            ModSub256(sx_gx, Gx[i], sxn);
//            ModSub256(px, _p2, sx_gx);
//            //ModSub256(px, Gx[i]);         // px = pow2(s) - p1.x - p2.x;
//
//            ModSub256(py, Gx[i], px);
//            _ModMult(py, _s);
//            ModSub256isOdd(py, Gy[i], &odd_py);
//
//            //ModSub256(py, Gy[i]);         // py = - p2.y - s*(ret.x-p2.x);
//            //odd_py = odd_py = py[0] & 1;
//
//
//            //CHECK_PREFIX(2048 / 2 + (i + 1));
//
//            _GetHash160Comp(px, odd_py, (uint8_t*)h);
//            CheckPoint(h, j * 1024 + 1024 / 2 + (i + 1), sPrefix, lookup32, out);
//
//            __syncthreads();
//
//            // P = StartPoint - i*G, if (x,y) = i*G then (x,-y) = -i*G
//            //Load256(px, sx);
//            ModSub256(dy, syn, Gy[i]);
//
//            _ModMult(_s, dy, dx[i]);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
//            _ModSqr(_p2, _s);             // _p = pow2(s)
//
//            ModSub256(px, _p2, sx_gx);
//            //ModSub256(px, Gx[i]);         // px = pow2(s) - p1.x - p2.x;
//
//            ModSub256(py, px, Gx[i]);
//            _ModMult(py, _s);             // py = s*(ret.x-p2.x)
//            ModSub256isOdd(py, Gy[i], py, &odd_py);// py = s*(ret.x-p2.x)
//
//
//            //ModSub256(py, Gy[i], py);     // py = - p2.y - s*(ret.x-p2.x);
//            //odd_py = py[0] & 1;
//
//
//
//            //CHECK_PREFIX(2048 / 2 - (i + 1));
//
//            _GetHash160Comp(px, odd_py, (uint8_t*)h);
//            CheckPoint(h, j * 1024 + 1024 / 2 - (i + 1), sPrefix, lookup32, out);
//
//        }
//
//        // First point (startP - (GRP_SZIE/2)*G)
//        __syncthreads();
//        //Load256(px, sx);
//        ModNeg256(dy, Gy[i]);
//        ModSub256(dy, sy);
//
//
//        _ModMult(_s, dy, dx[i]);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
//        _ModSqr(_p2, _s);              // _p = pow2(s)
//
//        ModSub256(px, _p2, sx);
//        ModSub256(px, Gx[i]);         // px = pow2(s) - p1.x - p2.x;
//
//        ModSub256(py, px, Gx[i]);
//        _ModMult(py, _s);
//        ModSub256isOdd(py, Gy[i], py, &odd_py);// py = s*(ret.x-p2.x)
//
//        //ModSub256(py, Gy[i], py);     // py = - p2.y - s*(ret.x-p2.x);
//        //odd_py = py[0] & 1;
//
//        //CHECK_PREFIX(0);
//
//        _GetHash160Comp(px, odd_py, (uint8_t*)h);
//        CheckPoint(h, j * 1024, sPrefix, lookup32, out);
//
//        i++;
//
//        // Next start point (startP + 2048*G)
//        __syncthreads();
//        /*Load256(px, sx);
//        Load256(py, sy);*/
//        ModSub256(dy, _2Gny, sy);
//
//        _ModMult(_s, dy, dx[i]);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
//        _ModSqr(_p2, _s);             // _p2 = pow2(s)
//
//        ModSub256(px, _p2, sx);
//        ModSub256(px, _2Gnx);         // px = pow2(s) - p1.x - p2.x;
//
//        ModSub256(py, _2Gnx, px);
//        _ModMult(py, _s);             // py = - s*(ret.x-p2.x)
//        ModSub256(py, _2Gny);         // py = - p2.y - s*(ret.x-p2.x);
//
//
//        Load256(sx, px);
//        Load256(sy, py);
//
//    }
//
//    // Update starting point
//    __syncthreads();
//    Store256A(startx, px);
//    Store256A(starty, py);
//
//}

//__global__ void comp_keys_2048(prefix_t* sPrefix, uint32_t* lookup32, uint64_t* keys, uint32_t* out) {
//
//    uint64_t* startx = keys + (blockIdx.x * blockDim.x) * 8;
//    uint64_t* starty = keys + (blockIdx.x * blockDim.x) * 8 + 4 * blockDim.x;
//
//    uint64_t dx[2048 / 2 + 1][4];
//    uint64_t px[4];
//    uint64_t py[4];
//    uint64_t sxn[4];
//    uint64_t syn[4];
//    uint64_t sx_gx[4];
//    uint64_t sx[4];
//    uint64_t sy[4];
//    uint64_t dy[4];
//    uint64_t _s[4];
//    uint64_t _p2[4];
//    uint8_t odd_py;
//    uint32_t h[5];
//
//    //uint64_t _2Gnx[4] = { 0x0646E23FD5F51508ULL,0xD8C39CABD5AC1CA1ULL,0xEA2A6E3E172DE238ULL,0x8282263212C609D9ULL };  //256
//    //uint64_t _2Gny[4] = { 0xD31B6EAFF6E26CAFULL,0x62D613AC2F7B17BEULL,0x5E8256E830B60ACEULL,0x11F8A8098557DFE4ULL };
//
//    //uint64_t _2Gnx[4] = { 0xD5B901B2E285131FULL,0xAAEC6ECDC813B088ULL,0xD664A18F66AD6240ULL,0x241FEBB8E23CBD77ULL }; //1024
//    //uint64_t _2Gny[4] = { 0xABB3E66F2750026DULL,0xCD50FD0FBD0CB5AFULL,0xD6C420BD13981DF8ULL,0x513378D9FF94F8D3ULL };
//
//    uint64_t _2Gnx[4] = { 0xEDCB63069B920471ULL, 0xFC318B85F423DE0DULL, 0xFCE4CC2983D8F8D9ULL, 0x5D1BDB4EA172FA79ULL }; //2048
//    uint64_t _2Gny[4] = { 0x70330666F7B83103ULL, 0x79EB1E9996C56E7BULL, 0x794BB99438A22656ULL, 0x2843826779379E2EULL };
//
//
//    // Load starting key
//    __syncthreads();
//    Load256A(sx, startx);
//    Load256A(sy, starty);
//    Load256(px, sx);
//    //Load256(py, sy);
//    /*Load256(px, sx);
//    Load256(py, sy);*/
//
//    uint32_t i;
//    uint32_t j;
//
//    for (j = 0; j < STEP_SIZE / 2048; j++) {
//
//        // Check starting point
//        odd_py = sy[0] & 1;
//        //CHECK_PREFIX(2048 / 2);
//
//        _GetHash160Comp(px, odd_py, (uint8_t*)h);
//        CheckPoint(h, j * 2048 + 2048 / 2, sPrefix, lookup32, out);
//
//        // Fill group with delta x
//
//        for (i = 0; i < 2048/2; i++)
//            ModSub256(dx[i], Gx[i], sx);
//        ModSub256(dx[i], _2Gnx, sx);  // For the next center point
//
//        // Compute modular inverse
//        _ModInvGrouped2048(dx);
//
//        ModNeg256(syn, sy);
//        ModNeg256(sxn, sx);
//
//        for (i = 0; i < 2048/2-1; i++) {
//
//            __syncthreads();
//            // P = StartPoint + i*G
//            /*Load256(px, sx);
//            Load256(py, sy);*/
//
//            ModSub256(dy, Gy[i], sy);
//
//            _ModMult(_s, dy, dx[i]);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
//            _ModSqr(_p2, _s);             // _p2 = pow2(s)
//
//            ModSub256(sx_gx, Gx[i], sxn);
//            ModSub256(px, _p2, sx_gx);
//            //ModSub256(px, Gx[i]);         // px = pow2(s) - p1.x - p2.x;
//
//            ModSub256(py, Gx[i], px);
//            _ModMult(py, _s);
//            ModSub256isOdd(py, Gy[i], &odd_py);
//
//            //ModSub256(py, Gy[i]);         // py = - p2.y - s*(ret.x-p2.x);
//            //odd_py = odd_py = py[0] & 1;
//
//
//            //CHECK_PREFIX(2048 / 2 + (i + 1));
//
//            _GetHash160Comp(px, odd_py, (uint8_t*)h);
//            CheckPoint(h, j * 2048 + 2048 / 2 + (i + 1), sPrefix, lookup32, out);
//
//            __syncthreads();
//
//            // P = StartPoint - i*G, if (x,y) = i*G then (x,-y) = -i*G
//            //Load256(px, sx);
//            ModSub256(dy, syn, Gy[i]);
//
//            _ModMult(_s, dy, dx[i]);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
//            _ModSqr(_p2, _s);             // _p = pow2(s)
//
//            ModSub256(px, _p2, sx_gx);
//            //ModSub256(px, Gx[i]);         // px = pow2(s) - p1.x - p2.x;
//
//            ModSub256(py, px, Gx[i]);
//            _ModMult(py, _s);             // py = s*(ret.x-p2.x)
//            ModSub256isOdd(py, Gy[i], py, &odd_py);// py = s*(ret.x-p2.x)
//
//
//            //ModSub256(py, Gy[i], py);     // py = - p2.y - s*(ret.x-p2.x);
//            //odd_py = py[0] & 1;
//
//
//            //CHECK_PREFIX(2048 / 2 - (i + 1));
//
//            _GetHash160Comp(px, odd_py, (uint8_t*)h);
//            CheckPoint(h, j * 2048 + 2048 / 2 - (i + 1), sPrefix, lookup32, out);
//
//        }
//
//        // First point (startP - (GRP_SZIE/2)*G)
//        __syncthreads();
//        //Load256(px, sx);
//        ModNeg256(dy, Gy[i]);
//        ModSub256(dy, sy);
//
//
//        _ModMult(_s, dy, dx[i]);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
//        _ModSqr(_p2, _s);              // _p = pow2(s)
//
//        ModSub256(px, _p2, sx);
//        ModSub256(px, Gx[i]);         // px = pow2(s) - p1.x - p2.x;
//
//        ModSub256(py, px, Gx[i]);
//        _ModMult(py, _s);
//        ModSub256isOdd(py, Gy[i], py, &odd_py);// py = s*(ret.x-p2.x)
//
//        //ModSub256(py, Gy[i], py);     // py = - p2.y - s*(ret.x-p2.x);
//        //odd_py = py[0] & 1;
//
//        //CHECK_PREFIX(0);
//
//        _GetHash160Comp(px, odd_py, (uint8_t*)h);
//        CheckPoint(h, j * 2048, sPrefix, lookup32, out);
//
//        i++;
//
//        // Next start point (startP + 2048*G)
//        __syncthreads();
//        /*Load256(px, sx);
//        Load256(py, sy);*/
//        ModSub256(dy, _2Gny, sy);
//
//        _ModMult(_s, dy, dx[i]);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
//        _ModSqr(_p2, _s);             // _p2 = pow2(s)
//
//        ModSub256(px, _p2, sx);
//        ModSub256(px, _2Gnx);         // px = pow2(s) - p1.x - p2.x;
//
//        ModSub256(py, _2Gnx, px);
//        _ModMult(py, _s);             // py = - s*(ret.x-p2.x)
//        ModSub256(py, _2Gny);         // py = - p2.y - s*(ret.x-p2.x);
//
//
//        Load256(sx, px);
//        Load256(sy, py);
//
//    }
//
//    // Update starting point
//    __syncthreads();
//    Store256A(startx, px);
//    Store256A(starty, py);
//
//}


//__global__ void comp_keys_256(prefix_t* sPrefix, uint32_t* lookup32, uint64_t* keys, uint32_t* out) {
//
//    /*int xPtr = (blockIdx.x * blockDim.x) * 8;
//    int yPtr = xPtr + 4 * blockDim.x;*/
//
//    //uint64_t dx[256 / 2 + 1][4];
//    //uint64_t dx[256 / 2 + 1][4];
//    uint64_t dx[4];
//    uint64_t subp[256 / 2 + 1][4];
//    //uint64_t dxinv[4];
//
//    uint64_t* startx = keys + (blockIdx.x * blockDim.x) * 8;
//    uint64_t* starty = keys + (blockIdx.x * blockDim.x) * 8 + 4 * blockDim.x;
//
//    uint64_t _2Gnx[4] = { 0x0646E23FD5F51508ULL,0xD8C39CABD5AC1CA1ULL,0xEA2A6E3E172DE238ULL,0x8282263212C609D9ULL };  //256
//    uint64_t _2Gny[4] = { 0xD31B6EAFF6E26CAFULL,0x62D613AC2F7B17BEULL,0x5E8256E830B60ACEULL,0x11F8A8098557DFE4ULL };
//
//    uint64_t px[4];
//    uint64_t py[4];
//    uint64_t sxn[4];
//    uint64_t syn[4];
//    uint64_t sx[4];
//    uint64_t sy[4];
//    uint64_t sx_gx[4];
//    uint64_t dy[4];
//    uint64_t _s[4];
//    uint64_t _p2[4];
//    uint8_t odd_py;
//    uint32_t h[5];
//
//    uint64_t inverse[5];
//
//    // Load starting key
//
//
//    __syncthreads();
//    Load256A(sx, startx);
//    Load256A(sy, starty);
//    //Load256(px, sx);
//    //Load256(py, sy);
//    /*Load256(px, sx);
//    Load256(py, sy);*/
//
//    uint32_t i;
//    uint32_t j;
//
//    for (j = 0; j < 16384 / 256; j++) {
//
//
//        // Check starting point
//        odd_py = sy[0] & 1;
//        //CHECK_PREFIX(256 / 2);
//
//        _GetHash160Comp(sx, odd_py, (uint8_t*)h);
//        CheckPoint(h, j * 256 + 256 / 2, sPrefix, lookup32, out);
//
//        // Fill group with delta x
//
//
//        for (i = 0; i < 256 / 2; i++) {
//            ModSub256(subp[i], Gx[i], sx);
//            //Load256(dx[i],subp[i]);
//        }
//        ModSub256(subp[i], _2Gnx, sx);  // For the next center point
//        //Load256(dx[i], subp[i])
//        __syncthreads();
//
//
//        for (i = 256 / 2; i > 0; i--) {
//            _ModMult(subp[i - 1], subp[i], subp[i - 1]);
//        }
//        __syncthreads();
//
//        // We need 320bit signed int for ModInv
//        Load256(inverse, subp[0]);
//        inverse[4] = 0;
//        _ModInv(inverse);
//
//        //_ModInvGrouped256_2(dx,subp,inverse);
//
//
//        ModNeg256(syn, sy);
//        ModNeg256(sxn, sx);
//
//        for (i = 0; i < (256 / 2 - 1); i++) {
//
//            __syncthreads();
//
//            // P = StartPoint + i*G
//            /*Load256(px, sx);
//            Load256(py, sy);*/
//
//            ModSub256(dy, Gy[i], sy);
//            _ModMult(dx, subp[i + 1], inverse);
//            _ModMult(_s, dy, dx);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
//            _ModSqr(_p2, _s);             // _p2 = pow2(s)
//
//            ModSub256(sx_gx, Gx[i], sxn);
//
//            ModSub256(px, _p2, sx_gx);
//            //ModSub256(px, Gx[i]);         // px = pow2(s) - p1.x - p2.x;
//
//            ModSub256(py, Gx[i], px);
//            _ModMult(py, _s);
//            ModSub256isOdd(py, Gy[i], &odd_py);
//
//            //CHECK_PREFIX(256 / 2 + (i + 1));
//
//            _GetHash160Comp(px, odd_py, (uint8_t*)h);
//            CheckPoint(h, j * 256 + 256 / 2 + (i + 1), sPrefix, lookup32, out);
//
//            __syncthreads();
//
//            // P = StartPoint - i*G, if (x,y) = i*G then (x,-y) = -i*G
//            //Load256(px, sx);
//            ModSub256(dy, syn, Gy[i]);
//
//            _ModMult(_s, dy, dx);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
//            _ModSqr(_p2, _s);             // _p = pow2(s)
//
//            ModSub256(px, _p2, sx_gx);
//            //ModSub256(px, Gx[i]);         // px = pow2(s) - p1.x - p2.x;
//
//            ModSub256(py, px, Gx[i]);
//            _ModMult(py, _s);             // py = s*(ret.x-p2.x)
//            ModSub256isOdd(py, Gy[i], py, &odd_py);// py = s*(ret.x-p2.x)
//
//            //ModSub256(py, Gy[i], py);     // py = - p2.y - s*(ret.x-p2.x);
//            //odd_py = py[0] & 1;
//
//
//
//            //CHECK_PREFIX(256 / 2 - (i + 1));
//
//            _GetHash160Comp(px, odd_py, (uint8_t*)h);
//            CheckPoint(h, j * 256 + 256 / 2 - (i + 1), sPrefix, lookup32, out);
//
//
//            ModSub256(dx, Gx[i], sx);
//            _ModMult(inverse, dx);
//
//        }
//
//        __syncthreads();
//
//        // First point (startP - (GRP_SZIE/2)*G)
//
//        //Load256(px, sx);
//        /*ModNeg256(dy, Gy[i]);
//        ModSub256(dy, sy);*/
//
//        ModSub256(dy, syn, Gy[i]);
//
//        _ModMult(dx, subp[i + 1], inverse);
//        _ModMult(_s, dy, dx);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
//        _ModSqr(_p2, _s);              // _p = pow2(s)
//
//        ModSub256(px, _p2, sx);
//        ModSub256(px, Gx[i]);         // px = pow2(s) - p1.x - p2.x;
//
//        ModSub256(py, px, Gx[i]);
//        _ModMult(py, _s);
//        ModSub256isOdd(py, Gy[i], py, &odd_py);// py = s*(ret.x-p2.x)
//
//        //ModSub256(py, Gy[i], py);     // py = - p2.y - s*(ret.x-p2.x);
//        //odd_py = py[0] & 1;
//
//        //CHECK_PREFIX(0);
//
//        _GetHash160Comp(px, odd_py, (uint8_t*)h);
//        CheckPoint(h, j * 256, sPrefix, lookup32, out);
//
//        //_ModMult(inverse, dx[i]);
//        ModSub256(dx, Gx[i], sx);
//        _ModMult(inverse, dx);
//
//        i++;
//
//        Load256(dx, inverse);
//
//        // Next start point (startP + 256*G)
//        __syncthreads();
//        /*Load256(px, sx);
//        Load256(py, sy);*/
//
//
//        ModSub256(dy, _2Gny, sy);
//
//        _ModMult(_s, dy, dx);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
//        _ModSqr(_p2, _s);             // _p2 = pow2(s)
//
//        ModSub256(px, _p2, sx);
//        ModSub256(px, _2Gnx);         // px = pow2(s) - p1.x - p2.x;
//
//        ModSub256(py, _2Gnx, px);
//        _ModMult(py, _s);             // py = - s*(ret.x-p2.x)
//        ModSub256(py, _2Gny);         // py = - p2.y - s*(ret.x-p2.x);
//
//
//        Load256(sx, px);
//        Load256(sy, py);
//
//    }
//
//    // Update starting point
//    __syncthreads();
//    Store256A(startx, px);
//    Store256A(starty, py);
//
//}

__global__ void comp_keys_1024(prefix_t* sPrefix, uint32_t* lookup32, uint64_t* keys, uint32_t* out) {

    /*int xPtr = (blockIdx.x * blockDim.x) * 8;
    int yPtr = xPtr + 4 * blockDim.x;*/

    //uint64_t dx[256 / 2 + 1][4];
    uint64_t dx[1024 / 2 + 1];
    uint64_t subp[1024 / 2 + 1][4];
    //uint64_t dxinv[4];

    uint64_t* startx = keys + (blockIdx.x * blockDim.x) * 8;
    uint64_t* starty = keys + (blockIdx.x * blockDim.x) * 8 + 4 * blockDim.x;


    //uint64_t _2Gnx[4] = { 0xD5B901B2E285131FULL,0xAAEC6ECDC813B088ULL,0xD664A18F66AD6240ULL,0x241FEBB8E23CBD77ULL }; //1024
    //uint64_t _2Gny[4] = { 0xABB3E66F2750026DULL,0xCD50FD0FBD0CB5AFULL,0xD6C420BD13981DF8ULL,0x513378D9FF94F8D3ULL };

    uint64_t px[4];
    uint64_t py[4];
    uint64_t sxn[4];
    uint64_t syn[4];
    uint64_t sx[4];
    uint64_t sy[4];
    uint64_t sx_gx[4];
    uint64_t dy[4];
    uint64_t _s[4];
    //uint64_t _p2[4];
    uint8_t odd_py;
    uint32_t h[5];

    uint64_t inverse[5];

    // Load starting key


    __syncthreads();
    Load256A(sx, startx);
    Load256A(sy, starty);
    //Load256(px, sx);
    //Load256(py, sy);
    /*Load256(px, sx);
    Load256(py, sy);*/

    uint32_t i;
    uint32_t j;

    for (j = 0; j < 16384 / 1024; j++) {


        // Check starting point
        odd_py = sy[0] & 1;
        //CHECK_PREFIX(256 / 2);

        _GetHash160Comp(sx, odd_py, (uint8_t*)h);
        CheckPoint(h, j * 1024 + 1024 / 2, sPrefix, lookup32, out);

        // Fill group with delta x


        for (i = 0; i < 1024 / 2; i++) {
            ModSub256(subp[i], Gx[i], sx);
            //Load256(dx[i], subp[i]);
        }
        ModSub256(subp[i], _2Gnx, sx);  // For the next center point
        //Load256(dx[i], subp[i])
        __syncthreads();


        for (i = 1024 / 2; i > 0; i--) {
            _ModMult(subp[i - 1], subp[i], subp[i - 1]);
        }
        __syncthreads();

        // We need 320bit signed int for ModInv
        Load256(inverse, subp[0]);
        inverse[4] = 0;
        _ModInv(inverse);

        //_ModInvGrouped256_2(dx,subp,inverse);


        ModNeg256(syn, sy);
        ModNeg256(sxn, sx);

        for (i = 0; i < (1024 / 2 - 1); i++) {

            __syncthreads();

            // P = StartPoint + i*G
            /*Load256(px, sx);
            Load256(py, sy);*/

            ModSub256(dy, Gy[i], sy);
            _ModMult(dx, subp[i + 1], inverse);
            _ModMult(_s, dy, dx);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
            //_ModSqr(_p2, _s);             // _p2 = pow2(s)
            _ModSqr(px, _s);             // _p2 = pow2(s)

            ModSub256(sx_gx, Gx[i], sxn);

            //ModSub256(px, _p2, sx_gx);
            ModSub256(px, sx_gx);
            //ModSub256(px, Gx[i]);         // px = pow2(s) - p1.x - p2.x;

            ModSub256(py, Gx[i], px);
            _ModMult(py, _s);
            ModSub256isOdd(py, Gy[i], &odd_py);

            //CHECK_PREFIX(256 / 2 + (i + 1));

            _GetHash160Comp(px, odd_py, (uint8_t*)h);
            CheckPoint(h, j * 1024 + 1024 / 2 + (i + 1), sPrefix, lookup32, out);

            __syncthreads();

            // P = StartPoint - i*G, if (x,y) = i*G then (x,-y) = -i*G
            //Load256(px, sx);
            ModSub256(dy, syn, Gy[i]);

            _ModMult(_s, dy, dx);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
            //_ModSqr(_p2, _s);             // _p = pow2(s)
            _ModSqr(px, _s);             // _p2 = pow2(s)

            //ModSub256(px, _p2, sx_gx);
            ModSub256(px, sx_gx);
            //ModSub256(px, Gx[i]);         // px = pow2(s) - p1.x - p2.x;

            ModSub256(py, px, Gx[i]);
            _ModMult(py, _s);             // py = s*(ret.x-p2.x)
            ModSub256isOdd(py, Gy[i], py, &odd_py);// py = s*(ret.x-p2.x)

            //ModSub256(py, Gy[i], py);     // py = - p2.y - s*(ret.x-p2.x);
            //odd_py = py[0] & 1;



            //CHECK_PREFIX(256 / 2 - (i + 1));

            _GetHash160Comp(px, odd_py, (uint8_t*)h);
            CheckPoint(h, j * 1024 + 1024 / 2 - (i + 1), sPrefix, lookup32, out);

            ModSub256(dx, Gx[i], sx);
            _ModMult(inverse, dx);

        }

        __syncthreads();


        // First point (startP - (GRP_SZIE/2)*G)

        //Load256(px, sx);
        /*ModNeg256(dy, Gy[i]);
        ModSub256(dy, sy);*/
        ModSub256(dy, syn, Gy[i]);

        _ModMult(dx, subp[i + 1], inverse);
        _ModMult(_s, dy, dx);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
        //_ModSqr(_p2, _s);              // _p = pow2(s)
        _ModSqr(px, _s);             // _p2 = pow2(s)

       // ModSub256(px, _p2, sx);
        ModSub256(px, sx);
        ModSub256(px, Gx[i]);         // px = pow2(s) - p1.x - p2.x;

        ModSub256(py, px, Gx[i]);
        _ModMult(py, _s);
        ModSub256isOdd(py, Gy[i], py, &odd_py);// py = s*(ret.x-p2.x)

        //ModSub256(py, Gy[i], py);     // py = - p2.y - s*(ret.x-p2.x);
        //odd_py = py[0] & 1;

        //CHECK_PREFIX(0);

        _GetHash160Comp(px, odd_py, (uint8_t*)h);
        CheckPoint(h, j * 1024, sPrefix, lookup32, out);

        ModSub256(dx, Gx[i], sx);
        _ModMult(inverse, dx);

        i++;

        Load256(dx, inverse);

        // Next start point (startP + 256*G)
        __syncthreads();
        /*Load256(px, sx);
        Load256(py, sy);*/


        ModSub256(dy, _2Gny, sy);

        _ModMult(_s, dy, dx);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
        //_ModSqr(_p2, _s);             // _p2 = pow2(s)
        _ModSqr(px, _s);             // _p2 = pow2(s)

        //ModSub256(px, _p2, sx);
        ModSub256(px, sx);
        ModSub256(px, _2Gnx);         // px = pow2(s) - p1.x - p2.x;

        ModSub256(py, _2Gnx, px);
        _ModMult(py, _s);             // py = - s*(ret.x-p2.x)
        ModSub256(py, _2Gny);         // py = - p2.y - s*(ret.x-p2.x);


        Load256(sx, px);
        Load256(sy, py);

    }

    // Update starting point
    __syncthreads();
    Store256A(startx, px);
    Store256A(starty, py);

}

//__global__ void comp_keys_2048(prefix_t* sPrefix, uint32_t* lookup32, uint64_t* keys, uint32_t* out) {
//
//    /*int xPtr = (blockIdx.x * blockDim.x) * 8;
//    int yPtr = xPtr + 4 * blockDim.x;*/
//
//    //uint64_t dx[256 / 2 + 1][4];
//    //uint64_t dx[2048 / 2 + 1][4];
//    uint64_t dx[4];
//    uint64_t subp[2048 / 2 + 1][4];
//    //uint64_t dxinv[4];
//
//    uint64_t* startx = keys + (blockIdx.x * blockDim.x) * 8;
//    uint64_t* starty = keys + (blockIdx.x * blockDim.x) * 8 + 4 * blockDim.x;
//
//
//    uint64_t _2Gnx[4] = { 0xEDCB63069B920471ULL, 0xFC318B85F423DE0DULL, 0xFCE4CC2983D8F8D9ULL, 0x5D1BDB4EA172FA79ULL }; //2048
//    uint64_t _2Gny[4] = { 0x70330666F7B83103ULL, 0x79EB1E9996C56E7BULL, 0x794BB99438A22656ULL, 0x2843826779379E2EULL };
//
//    uint64_t px[4];
//    uint64_t py[4];
//    uint64_t sxn[4];
//    uint64_t syn[4];
//    uint64_t sx[4];
//    uint64_t sy[4];
//    uint64_t sx_gx[4];
//    uint64_t dy[4];
//    uint64_t _s[4];
//    uint64_t _p2[4];
//    uint8_t odd_py;
//    uint32_t h[5];
//
//    uint64_t inverse[5];
//
//    // Load starting key
//
//
//    __syncthreads();
//    Load256A(sx, startx);
//    Load256A(sy, starty);
//    //Load256(px, sx);
//    //Load256(py, sy);
//    /*Load256(px, sx);
//    Load256(py, sy);*/
//
//    uint32_t i;
//    uint32_t j;
//
//    for (j = 0; j < 16384 / 2048; j++) {
//
//
//        // Check starting point
//        odd_py = sy[0] & 1;
//        //CHECK_PREFIX(256 / 2);
//
//        _GetHash160Comp(sx, odd_py, (uint8_t*)h);
//        CheckPoint(h, j * 2048 + 2048 / 2, sPrefix, lookup32, out);
//
//        // Fill group with delta x
//
//
//        for (i = 0; i < 2048 / 2; i++) {
//            ModSub256(subp[i], Gx[i], sx);
//            //Load256(dx[i], subp[i]);
//        }
//        ModSub256(subp[i], _2Gnx, sx);  // For the next center point
//        //Load256(dx[i], subp[i])
//        __syncthreads();
//
//
//        for (i = 2048 / 2; i > 0; i--) {
//            _ModMult(subp[i - 1], subp[i], subp[i - 1]);
//        }
//        __syncthreads();
//
//        // We need 320bit signed int for ModInv
//        Load256(inverse, subp[0]);
//        inverse[4] = 0;
//        _ModInv(inverse);
//
//        //_ModInvGrouped256_2(dx,subp,inverse);
//
//
//        ModNeg256(syn, sy);
//        ModNeg256(sxn, sx);
//
//        for (i = 0; i < (2048 / 2 - 1); i++) {
//
//            __syncthreads();
//
//            // P = StartPoint + i*G
//            /*Load256(px, sx);
//            Load256(py, sy);*/
//
//            ModSub256(dy, Gy[i], sy);
//            _ModMult(dx, subp[i + 1], inverse);
//            _ModMult(_s, dy, dx);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
//            _ModSqr(_p2, _s);             // _p2 = pow2(s)
//
//            ModSub256(sx_gx, Gx[i], sxn);
//
//            ModSub256(px, _p2, sx_gx);
//            //ModSub256(px, Gx[i]);         // px = pow2(s) - p1.x - p2.x;
//
//            ModSub256(py, Gx[i], px);
//            _ModMult(py, _s);
//            ModSub256isOdd(py, Gy[i], &odd_py);
//
//            //CHECK_PREFIX(256 / 2 + (i + 1));
//
//            _GetHash160Comp(px, odd_py, (uint8_t*)h);
//            CheckPoint(h, j * 2048 + 2048 / 2 + (i + 1), sPrefix, lookup32, out);
//
//            __syncthreads();
//
//            // P = StartPoint - i*G, if (x,y) = i*G then (x,-y) = -i*G
//            //Load256(px, sx);
//            ModSub256(dy, syn, Gy[i]);
//
//            _ModMult(_s, dy, dx);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
//            _ModSqr(_p2, _s);             // _p = pow2(s)
//
//            ModSub256(px, _p2, sx_gx);
//            //ModSub256(px, Gx[i]);         // px = pow2(s) - p1.x - p2.x;
//
//            ModSub256(py, px, Gx[i]);
//            _ModMult(py, _s);             // py = s*(ret.x-p2.x)
//            ModSub256isOdd(py, Gy[i], py, &odd_py);// py = s*(ret.x-p2.x)
//
//            //ModSub256(py, Gy[i], py);     // py = - p2.y - s*(ret.x-p2.x);
//            //odd_py = py[0] & 1;
//
//
//
//            //CHECK_PREFIX(256 / 2 - (i + 1));
//
//            _GetHash160Comp(px, odd_py, (uint8_t*)h);
//            CheckPoint(h, j * 2048 + 2048 / 2 - (i + 1), sPrefix, lookup32, out);
//
//
//            ModSub256(dx, Gx[i], sx);
//            _ModMult(inverse, dx);
//
//        }
//
//        __syncthreads();
//
//
//        // First point (startP - (GRP_SZIE/2)*G)
//
//        //Load256(px, sx);
//        /*ModNeg256(dy, Gy[i]);
//        ModSub256(dy, sy);*/
//        ModSub256(dy, syn, Gy[i]);
//
//        _ModMult(dx, subp[i + 1], inverse);
//        _ModMult(_s, dy, dx);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
//        _ModSqr(_p2, _s);              // _p = pow2(s)
//
//        ModSub256(px, _p2, sx);
//        ModSub256(px, Gx[i]);         // px = pow2(s) - p1.x - p2.x;
//
//        ModSub256(py, px, Gx[i]);
//        _ModMult(py, _s);
//        ModSub256isOdd(py, Gy[i], py, &odd_py);// py = s*(ret.x-p2.x)
//
//        //ModSub256(py, Gy[i], py);     // py = - p2.y - s*(ret.x-p2.x);
//        //odd_py = py[0] & 1;
//
//        //CHECK_PREFIX(0);
//
//        _GetHash160Comp(px, odd_py, (uint8_t*)h);
//        CheckPoint(h, j * 2048, sPrefix, lookup32, out);
//
//
//        ModSub256(dx, Gx[i], sx);
//        _ModMult(inverse, dx);
//        //_ModMult(inverse, dx[i]);
//
//        i++;
//
//        Load256(dx, inverse);
//
//        // Next start point (startP + 256*G)
//        __syncthreads();
//        /*Load256(px, sx);
//        Load256(py, sy);*/
//
//
//        ModSub256(dy, _2Gny, sy);
//
//        _ModMult(_s, dy, dx);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
//        _ModSqr(_p2, _s);             // _p2 = pow2(s)
//
//        ModSub256(px, _p2, sx);
//        ModSub256(px, _2Gnx);         // px = pow2(s) - p1.x - p2.x;
//
//        ModSub256(py, _2Gnx, px);
//        _ModMult(py, _s);             // py = - s*(ret.x-p2.x)
//        ModSub256(py, _2Gny);         // py = - p2.y - s*(ret.x-p2.x);
//
//
//        Load256(sx, px);
//        Load256(sy, py);
//
//    }
//
//    // Update starting point
//    __syncthreads();
//    Store256A(startx, px);
//    Store256A(starty, py);
//
//
//}
//
//__global__ void comp_keys_2048_8192(prefix_t* sPrefix, uint32_t* lookup32, uint64_t* keys, uint32_t* out) {
//
//    /*int xPtr = (blockIdx.x * blockDim.x) * 8;
//    int yPtr = xPtr + 4 * blockDim.x;*/
//
//    //uint64_t dx[256 / 2 + 1][4];
//    //uint64_t dx[2048 / 2 + 1][4];
//    uint64_t dx[4];
//    uint64_t subp[2048 / 2 + 1][4];
//    //uint64_t dxinv[4];
//
//    uint64_t* startx = keys + (blockIdx.x * blockDim.x) * 8;
//    uint64_t* starty = keys + (blockIdx.x * blockDim.x) * 8 + 4 * blockDim.x;
//
//
//    uint64_t _2Gnx[4] = { 0xEDCB63069B920471ULL, 0xFC318B85F423DE0DULL, 0xFCE4CC2983D8F8D9ULL, 0x5D1BDB4EA172FA79ULL }; //2048
//    uint64_t _2Gny[4] = { 0x70330666F7B83103ULL, 0x79EB1E9996C56E7BULL, 0x794BB99438A22656ULL, 0x2843826779379E2EULL };
//
//    uint64_t px[4];
//    uint64_t py[4];
//    uint64_t sxn[4];
//    uint64_t syn[4];
//    uint64_t sx[4];
//    uint64_t sy[4];
//    uint64_t sx_gx[4];
//    uint64_t dy[4];
//    uint64_t _s[4];
//    uint64_t _p2[4];
//    uint8_t odd_py;
//    uint32_t h[5];
//
//    uint64_t inverse[5];
//
//    // Load starting key
//
//
//    __syncthreads();
//    Load256A(sx, startx);
//    Load256A(sy, starty);
//    Load256(px, sx);
//    //Load256(py, sy);
//    /*Load256(px, sx);
//    Load256(py, sy);*/
//
//    uint32_t i;
//    uint32_t j;
//
//    for (j = 0; j < 8192 / 2048; j++) {
//
//
//        // Check starting point
//        odd_py = sy[0] & 1;
//        //CHECK_PREFIX(256 / 2);
//
//        _GetHash160Comp(px, odd_py, (uint8_t*)h);
//        CheckPoint(h, j * 2048 + 2048 / 2, sPrefix, lookup32, out);
//
//        // Fill group with delta x
//
//
//        for (i = 0; i < 2048 / 2; i++) {
//            ModSub256(subp[i], Gx[i], sx);
//            //Load256(dx[i], subp[i]);
//        }
//        ModSub256(subp[i], _2Gnx, sx);  // For the next center point
//        //Load256(dx[i], subp[i])
//        __syncthreads();
//
//
//        for (i = 2048 / 2; i > 0; i--) {
//            _ModMult(subp[i - 1], subp[i], subp[i - 1]);
//        }
//        __syncthreads();
//
//        // We need 320bit signed int for ModInv
//        Load256(inverse, subp[0]);
//        inverse[4] = 0;
//        _ModInv(inverse);
//
//        //_ModInvGrouped256_2(dx,subp,inverse);
//
//
//        ModNeg256(syn, sy);
//        ModNeg256(sxn, sx);
//
//        for (i = 0; i < (2048 / 2 - 1); i++) {
//
//            __syncthreads();
//
//            // P = StartPoint + i*G
//            /*Load256(px, sx);
//            Load256(py, sy);*/
//
//            ModSub256(dy, Gy[i], sy);
//            _ModMult(dx, subp[i + 1], inverse);
//            _ModMult(_s, dy, dx);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
//            _ModSqr(_p2, _s);             // _p2 = pow2(s)
//
//            ModSub256(sx_gx, Gx[i], sxn);
//
//            ModSub256(px, _p2, sx_gx);
//            //ModSub256(px, Gx[i]);         // px = pow2(s) - p1.x - p2.x;
//
//            ModSub256(py, Gx[i], px);
//            _ModMult(py, _s);
//            ModSub256isOdd(py, Gy[i], &odd_py);
//
//            //CHECK_PREFIX(256 / 2 + (i + 1));
//
//            _GetHash160Comp(px, odd_py, (uint8_t*)h);
//            CheckPoint(h, j * 2048 + 2048 / 2 + (i + 1), sPrefix, lookup32, out);
//
//            __syncthreads();
//
//            // P = StartPoint - i*G, if (x,y) = i*G then (x,-y) = -i*G
//            //Load256(px, sx);
//            ModSub256(dy, syn, Gy[i]);
//
//            _ModMult(_s, dy, dx);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
//            _ModSqr(_p2, _s);             // _p = pow2(s)
//
//            ModSub256(px, _p2, sx_gx);
//            //ModSub256(px, Gx[i]);         // px = pow2(s) - p1.x - p2.x;
//
//            ModSub256(py, px, Gx[i]);
//            _ModMult(py, _s);             // py = s*(ret.x-p2.x)
//            ModSub256isOdd(py, Gy[i], py, &odd_py);// py = s*(ret.x-p2.x)
//
//            //ModSub256(py, Gy[i], py);     // py = - p2.y - s*(ret.x-p2.x);
//            //odd_py = py[0] & 1;
//
//
//
//            //CHECK_PREFIX(256 / 2 - (i + 1));
//
//            _GetHash160Comp(px, odd_py, (uint8_t*)h);
//            CheckPoint(h, j * 2048 + 2048 / 2 - (i + 1), sPrefix, lookup32, out);
//
//
//            ModSub256(dx, Gx[i], sx);
//            _ModMult(inverse, dx);
//
//        }
//
//        __syncthreads();
//
//
//        // First point (startP - (GRP_SZIE/2)*G)
//
//        //Load256(px, sx);
//        /*ModNeg256(dy, Gy[i]);
//        ModSub256(dy, sy);*/
//        ModSub256(dy, syn, Gy[i]);
//
//        _ModMult(dx, subp[i + 1], inverse);
//        _ModMult(_s, dy, dx);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
//        _ModSqr(_p2, _s);              // _p = pow2(s)
//
//        ModSub256(px, _p2, sx);
//        ModSub256(px, Gx[i]);         // px = pow2(s) - p1.x - p2.x;
//
//        ModSub256(py, px, Gx[i]);
//        _ModMult(py, _s);
//        ModSub256isOdd(py, Gy[i], py, &odd_py);// py = s*(ret.x-p2.x)
//
//        //ModSub256(py, Gy[i], py);     // py = - p2.y - s*(ret.x-p2.x);
//        //odd_py = py[0] & 1;
//
//        //CHECK_PREFIX(0);
//
//        _GetHash160Comp(px, odd_py, (uint8_t*)h);
//        CheckPoint(h, j * 2048, sPrefix, lookup32, out);
//
//
//        ModSub256(dx, Gx[i], sx);
//        _ModMult(inverse, dx);
//        //_ModMult(inverse, dx[i]);
//
//        i++;
//
//        Load256(dx, inverse);
//
//        // Next start point (startP + 256*G)
//        __syncthreads();
//        /*Load256(px, sx);
//        Load256(py, sy);*/
//
//
//        ModSub256(dy, _2Gny, sy);
//
//        _ModMult(_s, dy, dx);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
//        _ModSqr(_p2, _s);             // _p2 = pow2(s)
//
//        ModSub256(px, _p2, sx);
//        ModSub256(px, _2Gnx);         // px = pow2(s) - p1.x - p2.x;
//
//        ModSub256(py, _2Gnx, px);
//        _ModMult(py, _s);             // py = - s*(ret.x-p2.x)
//        ModSub256(py, _2Gny);         // py = - p2.y - s*(ret.x-p2.x);
//
//
//        Load256(sx, px);
//        Load256(sy, py);
//
//    }
//
//    // Update starting point
//    __syncthreads();
//    Store256A(startx, px);
//    Store256A(starty, py);
//
//
//}


// ---------------------------------------------------------------------------------------
int NB_TRHEAD_PER_GROUP;

using namespace std;

std::string toHex(unsigned char* data, int length) {

    string ret;
    char tmp[3];
    for (int i = 0; i < length; i++) {
        if (i && i % 4 == 0) ret.append(" ");
        sprintf(tmp, "%02hhX", (int)data[i]);
        ret.append(tmp);
    }
    return ret;

}

int _ConvertSMVer2Cores(int major, int minor) {

    // Defines for GPU Architecture types (using the SM version to determine
    // the # of cores per SM
    typedef struct {
        int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
        // and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x60,  64},
        {0x61, 128},
        {0x62, 128},
        {0x70,  64},
        {0x72,  64},
        {0x75,  64},
        {0x80,  64},
        {0x86,  128},
        {0x89,  128},
        {-1, -1} };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    return 0;

}


int g_gpuId;
std::string globalGPUname;

GPUEngine::GPUEngine(int nbThreadGroup, int gpuId, uint32_t maxFound, bool rekey) {

    g_gpuId = gpuId;

    // Initialise CUDA
    this->rekey = rekey;
    initialised = false;
    cudaError_t err;

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf("GPUEngine: CudaGetDeviceCount %s\n", cudaGetErrorString(error_id));
        return;
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0) {
        printf("GPUEngine: There are no available device(s) that support CUDA\n");
        return;
    }

    err = cudaSetDevice(gpuId);
    if (err != cudaSuccess) {
        printf("GPUEngine: %s\n", cudaGetErrorString(err));
        return;
    }

    //// Andrew mod
    //   set cpu spinwait flag to prevent 100% cpu usage
    //err = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    //if (err != cudaSuccess) {
    //    fprintf(stderr, "GPUEngine: %s\n", cudaGetErrorString(err));
    //    return;
    //}

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, gpuId);



    int major = deviceProp.major; // Compute Capability major
    int minor = deviceProp.minor; // Compute Capability minor

    int ver = 10 * major + minor;

    if (ver > 85) {
        NB_TRHEAD_PER_GROUP = 256;
        nbThreadGroup = deviceProp.multiProcessorCount * 128;
    }
    else {
        NB_TRHEAD_PER_GROUP = 256;
        nbThreadGroup = deviceProp.multiProcessorCount * 128;
    }


    uint64_t powerOfTwo = 1;
    while (powerOfTwo <= nbThreadGroup) {  ///<=
        powerOfTwo <<= 1;
    }

    powerOfTwo >>= 1;
    nbThreadGroup = powerOfTwo;

    int cput = omp_get_max_threads();

    if (nbThreadGroup >= 16384) {
        if (cput < 8) {
            nbThreadGroup = nbThreadGroup / 2;
        }
    }


    this->nbThread = nbThreadGroup * NB_TRHEAD_PER_GROUP;//////////////////////////////////////////////////////////////////
    this->maxFound = maxFound;
    this->outputSize = (maxFound * ITEM_SIZE + 4);

    char tmp[512];
    /*sprintf(tmp,"GPU #%d %s (%dx%d cores) Grid(%dx%d)",
    gpuId,deviceProp.name,deviceProp.multiProcessorCount,
    _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
    nbThread / NB_TRHEAD_PER_GROUP,
    NB_TRHEAD_PER_GROUP);*/
    sprintf(tmp, "GPU #%d %s",
        gpuId, deviceProp.name);

    deviceName = std::string(tmp);

    globalGPUname = deviceProp.name;

    // Prefer L1 (We do not use __shared__ at all)
    err = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    if (err != cudaSuccess) {
        printf("GPUEngine: %s\n", cudaGetErrorString(err));
        return;
    }

    //size_t stackSize = 49152;
    //err = cudaDeviceSetLimit(cudaLimitStackSize, stackSize);
    //if (err != cudaSuccess) {
    //  printf("GPUEngine: %s\n", cudaGetErrorString(err));
    //  return;
    //}

    /*
    size_t heapSize = ;
    err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize);
    if (err != cudaSuccess) {
      printf("Error: %s\n", cudaGetErrorString(err));
      exit(0);
    }

    size_t size;
    cudaDeviceGetLimit(&size, cudaLimitStackSize);
    printf("Stack Size %lld\n", size);
    cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
    printf("Heap Size %lld\n", size);
    */

    // Allocate memory
    err = cudaMalloc((void**)&inputPrefix, _64K * 2);
    if (err != cudaSuccess) {
        printf("GPUEngine: Allocate prefix memory: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaHostAlloc(&inputPrefixPinned, _64K * 2, cudaHostAllocWriteCombined | cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("GPUEngine: Allocate prefix pinned memory: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMalloc((void**)&inputKey, nbThread * 32 * 2);
    if (err != cudaSuccess) {
        printf("GPUEngine: Allocate input memory: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaHostAlloc(&inputKeyPinned, nbThread * 32 * 2, cudaHostAllocWriteCombined | cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("GPUEngine: Allocate input pinned memory: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMalloc((void**)&outputPrefix, outputSize);
    if (err != cudaSuccess) {
        printf("GPUEngine: Allocate output memory: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaHostAlloc(&outputPrefixPinned, outputSize, cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("GPUEngine: Allocate output pinned memory: %s\n", cudaGetErrorString(err));
        return;
    }

    searchMode = SEARCH_COMPRESSED;
    searchType = P2PKH;
    initialised = true;
    pattern = "";
    hasPattern = false;
    inputPrefixLookUp = NULL;

}

void GPUEngine::PrintCudaInfo() {

    cudaError_t err;

    //const char *sComputeMode[] =
    //{
    //  "Multiple host threads",
    //  "Only one host thread",
    //  "No host thread",
    //  "Multiple process threads",
    //  "Unknown",
    //   NULL
    //};

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    //if (error_id != cudaSuccess) {
    //  printf("GPUEngine: CudaGetDeviceCount %s\n", cudaGetErrorString(error_id));
    //  return;
    //}

    //// This function call returns 0 if there are no CUDA capable devices.
    //if (deviceCount == 0) {
    //  printf("GPUEngine: There are no available device(s) that support CUDA\n");
    //  return;
    //}

    for (int i = 0;i < deviceCount;i++) {

        /*err = cudaSetDevice(i);
        if (err != cudaSuccess) {
          printf("GPUEngine: cudaSetDevice(%d) %s\n", i, cudaGetErrorString(err));
          return;
        }*/

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        /*printf("GPU #%d %s (%dx%d cores) (Cap %d.%d) (%.1f MB) (%s)\n",
          i,deviceProp.name,deviceProp.multiProcessorCount,
          _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
          deviceProp.major, deviceProp.minor,(double)deviceProp.totalGlobalMem/1048576.0,
          sComputeMode[deviceProp.computeMode]);*/

        printf("%d , %s", i, deviceProp.name);

    }

}

GPUEngine::~GPUEngine() {

    cudaFree(inputKey);
    cudaFree(inputPrefix);
    if (inputPrefixLookUp) cudaFree(inputPrefixLookUp);
    cudaFreeHost(outputPrefixPinned);
    cudaFree(outputPrefix);

}

int GPUEngine::GetNbThread() {
    return nbThread;
}

void GPUEngine::SetSearchMode(int searchMode) {
    this->searchMode = searchMode;
}

void GPUEngine::SetSearchType(int searchType) {
    this->searchType = searchType;
}

void GPUEngine::SetPrefix(std::vector<prefix_t> prefixes) {

    memset(inputPrefixPinned, 0, _64K * 2);
    for (int i = 0;i < (int)prefixes.size();i++)
        inputPrefixPinned[prefixes[i]] = 1;

    // Fill device memory
    cudaMemcpy(inputPrefix, inputPrefixPinned, _64K * 2, cudaMemcpyHostToDevice);

    // We do not need the input pinned memory anymore
    cudaFreeHost(inputPrefixPinned);
    inputPrefixPinned = NULL;
    lostWarning = false;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: SetPrefix: %s\n", cudaGetErrorString(err));
    }

}

void GPUEngine::SetPattern(const char* pattern) {

    strcpy((char*)inputPrefixPinned, pattern);

    // Fill device memory
    cudaMemcpy(inputPrefix, inputPrefixPinned, _64K * 2, cudaMemcpyHostToDevice);

    // We do not need the input pinned memory anymore
    cudaFreeHost(inputPrefixPinned);
    inputPrefixPinned = NULL;
    lostWarning = false;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: SetPattern: %s\n", cudaGetErrorString(err));
    }

    hasPattern = true;

}

void GPUEngine::SetPrefix(std::vector<LPREFIX> prefixes, uint32_t totalPrefix) {

    // Allocate memory for the second level of lookup tables
    cudaError_t err = cudaMalloc((void**)&inputPrefixLookUp, (_64K + totalPrefix) * 4);
    if (err != cudaSuccess) {
        printf("GPUEngine: Allocate prefix lookup memory: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaHostAlloc(&inputPrefixLookUpPinned, (_64K + totalPrefix) * 4, cudaHostAllocWriteCombined | cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("GPUEngine: Allocate prefix lookup pinned memory: %s\n", cudaGetErrorString(err));
        return;
    }

    uint32_t offset = _64K;
    memset(inputPrefixPinned, 0, _64K * 2);
    memset(inputPrefixLookUpPinned, 0, _64K * 4);
    for (int i = 0; i < (int)prefixes.size(); i++) {
        int nbLPrefix = (int)prefixes[i].lPrefixes.size();
        inputPrefixPinned[prefixes[i].sPrefix] = (uint16_t)nbLPrefix;
        inputPrefixLookUpPinned[prefixes[i].sPrefix] = offset;
        for (int j = 0; j < nbLPrefix; j++) {
            inputPrefixLookUpPinned[offset++] = prefixes[i].lPrefixes[j];
        }
    }

    if (offset != (_64K + totalPrefix)) {
        printf("GPUEngine: Wrong totalPrefix %d!=%d!\n", offset - _64K, totalPrefix);
        return;
    }

    // Fill device memory
    cudaMemcpy(inputPrefix, inputPrefixPinned, _64K * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(inputPrefixLookUp, inputPrefixLookUpPinned, (_64K + totalPrefix) * 4, cudaMemcpyHostToDevice);

    // We do not need the input pinned memory anymore
    cudaFreeHost(inputPrefixPinned);
    inputPrefixPinned = NULL;
    cudaFreeHost(inputPrefixLookUpPinned);
    inputPrefixLookUpPinned = NULL;
    lostWarning = false;

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: SetPrefix (large): %s\n", cudaGetErrorString(err));
    }

}


int GPUEngine::GetStepSize() {

    //cudaDeviceProp deviceProp;
    //cudaGetDeviceProperties(&deviceProp, g_gpuId);

    //int major = deviceProp.major; // Compute Capability major
    //int minor = deviceProp.minor; // Compute Capability minor

    //int ver = 10 * major + minor;

    //if (ver > 85) {
    //    return 16384;
    //}
    //else {
    //    return 8192;
    //}

    return 16384;

}

int GPUEngine::GetGroupSize() {
    //return GRP_SIZE;
    //cudaDeviceProp deviceProp;
    //cudaGetDeviceProperties(&deviceProp, g_gpuId);

    //int major = deviceProp.major; // Compute Capability major
    //int minor = deviceProp.minor; // Compute Capability minor

    //int ver = 10 * major + minor;

    //if (ver > 88) {
    //    return 1024;
    //}
    //else {
    //    return 2048;
    //}

    return 1024;

}


bool GPUEngine::callKernel() {

    // Reset nbFound
    cudaMemset(outputPrefix, 0, 4);


    //cudaDeviceProp deviceProp;
    //cudaGetDeviceProperties(&deviceProp, g_gpuId);

    //int major = deviceProp.major; // Compute Capability major
    //int minor = deviceProp.minor; // Compute Capability minor
    //int ver = 10 * major + minor;

    //if (ver > 88) {
    //    comp_keys_1024 << < nbThread / NB_TRHEAD_PER_GROUP, NB_TRHEAD_PER_GROUP >> >
    //        (inputPrefix, inputPrefixLookUp, inputKey, outputPrefix);
    //}
    //else {
    //    if (ver > 85) {
    //        comp_keys_2048 << < nbThread / NB_TRHEAD_PER_GROUP, NB_TRHEAD_PER_GROUP >> >
    //            (inputPrefix, inputPrefixLookUp, inputKey, outputPrefix);
    //    }
    //    else {
    //        comp_keys_2048_8192 << < nbThread / NB_TRHEAD_PER_GROUP, NB_TRHEAD_PER_GROUP >> >
    //            (inputPrefix, inputPrefixLookUp, inputKey, outputPrefix);
    //    }
    //}

    comp_keys_1024 << < nbThread / NB_TRHEAD_PER_GROUP, NB_TRHEAD_PER_GROUP >> >
        (inputPrefix, inputPrefixLookUp, inputKey, outputPrefix);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: Kernel: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;

}



bool GPUEngine::SetKeys(Point* p) {

    // Sets the starting keys for each thread
    // p must contains nbThread public keys

    for (int i = 0; i < nbThread; i += NB_TRHEAD_PER_GROUP) {
        for (int j = 0; j < NB_TRHEAD_PER_GROUP; j++) {

            inputKeyPinned[8 * i + j + 0 * NB_TRHEAD_PER_GROUP] = p[i + j].x.bits64[0];
            inputKeyPinned[8 * i + j + 1 * NB_TRHEAD_PER_GROUP] = p[i + j].x.bits64[1];
            inputKeyPinned[8 * i + j + 2 * NB_TRHEAD_PER_GROUP] = p[i + j].x.bits64[2];
            inputKeyPinned[8 * i + j + 3 * NB_TRHEAD_PER_GROUP] = p[i + j].x.bits64[3];

            inputKeyPinned[8 * i + j + 4 * NB_TRHEAD_PER_GROUP] = p[i + j].y.bits64[0];
            inputKeyPinned[8 * i + j + 5 * NB_TRHEAD_PER_GROUP] = p[i + j].y.bits64[1];
            inputKeyPinned[8 * i + j + 6 * NB_TRHEAD_PER_GROUP] = p[i + j].y.bits64[2];
            inputKeyPinned[8 * i + j + 7 * NB_TRHEAD_PER_GROUP] = p[i + j].y.bits64[3];

        }
    }

    // Fill device memory

    cudaMemcpy(inputKey, inputKeyPinned, nbThread * 32 * 2, cudaMemcpyHostToDevice);
    // We do not need the input pinned memory anymore
    cudaFreeHost(inputKeyPinned);
    inputKeyPinned = NULL;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: SetKeys: %s\n", cudaGetErrorString(err));
    }

    return callKernel();

}



bool GPUEngine::Launch(std::vector<ITEM>& prefixFound, bool spinWait) {


    prefixFound.clear();

    // Get the result


    if(spinWait) {

      cudaMemcpy(outputPrefixPinned, outputPrefix, outputSize, cudaMemcpyDeviceToHost);

    } else {

      // Use cudaMemcpyAsync to avoid default spin wait of cudaMemcpy wich takes 100% CPU
      cudaEvent_t evt;
      cudaEventCreate(&evt);

      //cudaMemcpy(outputPrefixPinned, outputPrefix, 4, cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(outputPrefixPinned, outputPrefix, 4, cudaMemcpyDeviceToHost, 0);

      cudaEventRecord(evt, 0);
      while (cudaEventQuery(evt) == cudaErrorNotReady) {
        // Sleep 1 ms to free the CPU
        Timer::SleepMillis(1);
      }
      cudaEventDestroy(evt);

    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("GPUEngine: Launch: %s\n", cudaGetErrorString(err));
      return false;
    }

    // Look for prefix found
    uint32_t nbFound = outputPrefixPinned[0];

    if (nbFound > maxFound) {
      // prefix has been lost
      if (!lostWarning) {
        printf("\nWarning, %d items lost\nHint: Search with less prefixes, less threads (-g) or increase maxFound (-m)\n", (nbFound - maxFound));
        lostWarning = true;
      }
      nbFound = maxFound;
    }

    // When can perform a standard copy, the kernel is eneded
    cudaMemcpy(outputPrefixPinned, outputPrefix, nbFound * ITEM_SIZE + 4, cudaMemcpyDeviceToHost);

    for (uint32_t i = 0; i < nbFound; i++) {
        uint32_t* itemPtr = outputPrefixPinned + (i * ITEM_SIZE32 + 1);
        ITEM it;
        it.thId = itemPtr[0];
        int16_t* ptr = (int16_t*)&(itemPtr[1]);
        it.endo = ptr[0] & 0x7FFF;
        it.mode = (ptr[0] & 0x8000) != 0;
        it.incr = ptr[1];
        it.hash = (uint8_t*)(itemPtr + 2);
        prefixFound.push_back(it);
    }

    return callKernel();

}
