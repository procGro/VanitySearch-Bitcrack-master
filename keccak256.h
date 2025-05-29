#ifndef KECCAK256_H
#define KECCAK256_H

// Computes the Keccak-256 hash of the input.
// input: pointer to the input data.
// input_len: length of the input data in bytes.
// output: pointer to a 32-byte array where the hash will be stored.
void keccak256(const unsigned char *input, unsigned int input_len, unsigned char *output);

#endif // KECCAK256_H
