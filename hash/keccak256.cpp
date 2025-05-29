/* 
 * Bitcoin cryptography library
 * Copyright (c) Project Nayuki
 * 
 * https://www.nayuki.io/page/bitcoin-cryptography-library
 * https://github.com/nayuki/Bitcoin-Cryptography-Library
 */

#include <algorithm>
#include <cstring>
#include "Keccak256.hpp"

using std::uint8_t;
using std::uint64_t;
using std::size_t;


// Public function
void Keccak256::getHash(const uint8_t msg[], size_t len, uint8_t hashResult[HASH_LEN]) {
	uint64_t state[5][5] = {};
	
	// Absorb
	size_t off = 0;
	while (off < len) {
		size_t count = std::min(len - off, static_cast<size_t>(BLOCK_SIZE));
		for (size_t i = 0; i < count; i++)
			state[(i % 5)][(i / 5)] ^= static_cast<uint64_t>(msg[off + i]) << ((i % BLOCK_SIZE) % 8 * 8);
		off += count;
		if (count == BLOCK_SIZE)
			absorb(state);
	}
	
	// Pad
	state[(off % BLOCK_SIZE / 8 % 5)][(off % BLOCK_SIZE / 8 / 5)] ^= UINT64_C(0x01) << ((off % BLOCK_SIZE % 8) * 8);
	off = BLOCK_SIZE - 1;
	state[(off % BLOCK_SIZE / 8 % 5)][(off % BLOCK_SIZE / 8 / 5)] ^= UINT64_C(0x80) << ((off % BLOCK_SIZE % 8) * 8);
	absorb(state);
	
	// Squeeze
	for (int i = 0; i < HASH_LEN; i++)
		hashResult[i] = static_cast<uint8_t>(state[i % BLOCK_SIZE / 8 % 5][i % BLOCK_SIZE / 8 / 5] >> ((i % BLOCK_SIZE % 8) * 8));
}


// Private function
void Keccak256::absorb(uint64_t state[5][5]) {
	uint64_t C[5];
	uint64_t D[5];
	uint64_t B[5][5];
	
	for (int r = 0; r < NUM_ROUNDS; r++) {
		// Theta step
		for (int x = 0; x < 5; x++) {
			C[x] = 0;
			for (int y = 0; y < 5; y++)
				C[x] ^= state[x][y];
		}
		for (int x = 0; x < 5; x++) {
			D[x] = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
			for (int y = 0; y < 5; y++)
				state[x][y] ^= D[x];
		}
		
		// Rho and pi steps
		uint64_t temp = state[1][0];
		for (int x = 0; x < 5; x++) {
			for (int y = 0; y < 5; y++) {
				int x2 = y;
				int y2 = (x * 2 + y * 3) % 5;
				B[x2][y2] = rotl64(state[x][y], ROTATION[x][y]);
			}
		}
		state[1][0] = B[1][0];  // To be overwritten by Chi step
		
		// Chi step
		for (int x = 0; x < 5; x++) {
			for (int y = 0; y < 5; y++)
				state[x][y] = B[x][y] ^ (~B[(x + 1) % 5][y] & B[(x + 2) % 5][y]);
		}
		
		// Iota step
		uint64_t rc = 0;
		switch (r) {
			case  0: rc = UINT64_C(0x0000000000000001); break;
			case  1: rc = UINT64_C(0x0000000000008082); break;
			case  2: rc = UINT64_C(0x800000000000808A); break;
			case  3: rc = UINT64_C(0x8000000080008000); break;
			case  4: rc = UINT64_C(0x000000000000808B); break;
			case  5: rc = UINT64_C(0x0000000080000001); break;
			case  6: rc = UINT64_C(0x8000000080008081); break;
			case  7: rc = UINT64_C(0x8000000000008009); break;
			case  8: rc = UINT64_C(0x000000000000008A); break;
			case  9: rc = UINT64_C(0x0000000000000088); break;
			case 10: rc = UINT64_C(0x0000000080008009); break;
			case 11: rc = UINT64_C(0x000000008000000A); break;
			case 12: rc = UINT64_C(0x000000008000808B); break;
			case 13: rc = UINT64_C(0x800000000000008B); break;
			case 14: rc = UINT64_C(0x8000000000008089); break;
			case 15: rc = UINT64_C(0x8000000000008003); break;
			case 16: rc = UINT64_C(0x8000000000008002); break;
			case 17: rc = UINT64_C(0x8000000000000080); break;
			case 18: rc = UINT64_C(0x000000000000800A); break;
			case 19: rc = UINT64_C(0x800000008000000A); break;
			case 20: rc = UINT64_C(0x8000000080008081); break;
			case 21: rc = UINT64_C(0x8000000000008080); break;
			case 22: rc = UINT64_C(0x0000000080000001); break;
			case 23: rc = UINT64_C(0x8000000080008008); break;
			default: throw "Unreachable";  // Should not happen
		}
		state[0][0] ^= rc;
	}
}


// Private data
const unsigned char Keccak256::ROTATION[5][5] = {
	{ 0, 36,  3, 41, 18},
	{ 1, 44, 10, 45,  2},
	{62,  6, 43, 15, 61},
	{28, 55, 25, 21, 56},
	{27, 20, 39,  8, 14},
};


// Private function
uint64_t Keccak256::rotl64(uint64_t x, int i) {
	return (x << i) | (x >> (64 - i));
}
