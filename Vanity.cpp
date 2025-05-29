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

#include "Vanity.h"
#include "Base58.h"
#include "Bech32.h"
#include "hash/sha256.h"
#include "hash/sha512.h"
#include "IntGroup.h"
#include "Wildcard.h"
#include "Timer.h"
#include "hash/ripemd160.h"
#include <string.h>
#include <math.h>
#include <algorithm>
#include <thread>
#include <atomic>
#include <ctime>

#ifdef WIN64
#include <windows.h>
#include <process.h>
#endif

//#define GRP_SIZE 256

using namespace std;

//Point Gn[GRP_SIZE / 2];
//Point _2Gn;

VanitySearch::VanitySearch(Secp256K1* secp, vector<std::string>& inputAddresses, int searchMode,
	bool stop, string outputFile, uint32_t maxFound, BITCRACK_PARAM* bc):inputAddresses(inputAddresses) 
{
	this->secp = secp;
	this->searchMode = searchMode;
	this->stopWhenFound = stop;
	this->outputFile = outputFile;
	this->numGPUs = 0;
	this->maxFound = maxFound;	
	this->searchType = -1;
	this->bc = bc;	
	this->threads = nullptr;

	rseed(time(NULL));
	
	addresses.clear();

	// Create a 65536 items lookup table
	// For Ethereum, this table will not be effectively used by the CPU path,
	// as matching relies on iterating through `this->inputAddresses`.
	ADDRESS_TABLE_ITEM t;
	t.found = true;
	t.items = NULL;
	for (int i = 0; i < 65536; i++)
		addresses.push_back(t);
	
	// Insert addresses
	bool loadingProgress = (inputAddresses.size() > 1000);
	if (loadingProgress)
		fprintf(stdout, "[Building lookup16   0.0%%]\r");

	nbAddress = 0;
	onlyFull = true;

	printf("Debug: searchMode = %d, searchType = %d\n", searchMode, searchType);

	for (int i = 0; i < (int)inputAddresses.size(); i++) 
	{
		ADDRESS_ITEM it;
		// For ETHEREUM, itAddresses vector is not really used in the same way,
		// as sAddress/lAddress based lookup is bypassed.
		// We still populate `it` to count `nbAddress`.
		std::vector<ADDRESS_ITEM> itAddresses; 

		// inputAddresses[i] is a reference to the string in the member vector.
		// initAddress will modify it in-place for ETHEREUM mode (lowercase).
		if (initAddress(inputAddresses[i], &it)) { 
			bool* found = new bool;
			*found = false;
			it.found = found;
			// For Ethereum, it.address now points to the (now lowercased) string
			// within the inputAddresses member vector itself.
			itAddresses.push_back(it); 
		}

		if (itAddresses.size() > 0) 
		{
			// Add the item to all correspoding addresses in the lookup table
			// This part is less relevant for ETHEREUM CPU search as it doesn't use the sAddress lookup table.
			if (this->searchType != ETHEREUM) { // Use this->searchType
				for (int j = 0; j < (int)itAddresses.size(); j++) 
				{
					address_t p = itAddresses[j].sAddress;

					if (addresses[p].items == NULL) {
						addresses[p].items = new vector<ADDRESS_ITEM>();
						addresses[p].found = false;
						usedAddress.push_back(p);
					}
					(*addresses[p].items).push_back(itAddresses[j]);
				}
			}
			onlyFull &= it.isFull; // isFull is set correctly for Ethereum in initAddress
			nbAddress++;
		}

		if (loadingProgress && i % 1000 == 0)
			fprintf(stdout, "[Building lookup16 %5.1f%%]\r", (((double)i) / (double)(inputAddresses.size() - 1)) * 100.0);
	}

	if (loadingProgress)
		fprintf(stdout, "\n");

	// Debug print removed as it was causing issues with static array searchModes
	// printf("Debug: Final searchType = %d, nbAddress = %d\n", searchType, nbAddress);

	if (nbAddress == 0) 
	{
		fprintf(stderr, "[ERROR] VanitySearch: nothing to search !\n");
		exit(-1);
	}

	// Second level lookup (less relevant for ETHEREUM CPU search)
	uint32_t unique_sAddress = 0;
	uint32_t minI = 0xFFFFFFFF;
	uint32_t maxI = 0;
	if (this->searchType != ETHEREUM) { // Use this->searchType
		for (int i = 0; i < (int)addresses.size(); i++) 
		{
			if (addresses[i].items) 
			{
				LADDRESS lit;
				lit.sAddress = i;
				if (addresses[i].items) 
				{
					for (int j = 0; j < (int)addresses[i].items->size(); j++) 
					{
						lit.lAddresses.push_back((*addresses[i].items)[j].lAddress);
					}
				}

				sort(lit.lAddresses.begin(), lit.lAddresses.end());
				usedAddressL.push_back(lit);
				if ((uint32_t)lit.lAddresses.size() > maxI) maxI = (uint32_t)lit.lAddresses.size();
				if ((uint32_t)lit.lAddresses.size() < minI) minI = (uint32_t)lit.lAddresses.size();
				unique_sAddress++;
			}

			if (loadingProgress) // This progress might be confusing if ETHEREUM is active
				fprintf(stdout, "[Building lookup32 %.1f%%]\r", ((double)i * 100.0) / (double)addresses.size());
		}
		if (loadingProgress)
			fprintf(stdout, "\n");
	}
	
	static const char* searchModes[] = {"P2PKH", "P2SH", "BECH32", "HASH160", "ETHEREUM"};
	string searchInfo = string(searchModes[this->searchMode]); // Use this->searchMode
	if (nbAddress < 10) 
	{	
		for (unsigned int i = 0; i < nbAddress; i++) // Use unsigned int for loop counter with nbAddress
		{
			// For Ethereum, inputAddresses[i] was already lowercased by initAddress
			fprintf(stdout, "Search: %s [%s]\n", inputAddresses[i].c_str(), searchInfo.c_str());
		}
	}
	else 
	{		
		if (this->searchType == ETHEREUM) { // Use this->searchType
			fprintf(stdout, "Search: %d Ethereum prefixes [%s]\n", nbAddress, searchInfo.c_str());
		} else {
			fprintf(stdout, "Search: %d (Lookup size %d,[%d,%d]) [%s]\n", nbAddress, unique_sAddress, minI, maxI, searchInfo.c_str());
		}
	}

	//// Compute Generator table G[n] = (n+1)*G
	//Point g = secp->G;
	//Gn[0] = g;
	//g = secp->DoubleDirect(g);
	//Gn[1] = g;
	//for (int i = 2; i < GRP_SIZE / 2; i++) {
	//	g = secp->AddDirect(g, secp->G);
	//	Gn[i] = g;
	//}
	//// _2Gn = CPU_GRP_SIZE*G
	//_2Gn = secp->DoubleDirect(Gn[GRP_SIZE / 2 - 1]);

	// Constant for endomorphism
	// if a is a nth primitive root of unity, a^-1 is also a nth primitive root.
	// beta^3 = 1 mod p implies also beta^2 = beta^-1 mop (by multiplying both side by beta^-1)
	// (beta^3 = 1 mod p),  beta2 = beta^-1 = beta^2
	// (lambda^3 = 1 mod n), lamba2 = lamba^-1 = lamba^2
	beta.SetBase16("7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee");
	lambda.SetBase16("5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72");
	beta2.SetBase16("851695d49a83f8ef919bb86153cbcb16630fb68aed0a766a3ec693d68e6afa40");
	lambda2.SetBase16("ac9c52b33fa3cf1f5ad9e3fd77ed9ba4a880b9fc8ec739c2e0cfc810b51283ce");

	startKey.Set(&bc->ksNext);	

	char* ctimeBuff;
	time_t now = time(NULL);
	ctimeBuff = ctime(&now);
	fprintf(stdout, "Current task START time: %s", ctimeBuff);
	fflush(stdout);
}

bool VanitySearch::isSingularAddress(std::string pref) {

	// check is the given address contains only 1
	bool only1 = true;
	int i = 0;
	while (only1 && i < (int)pref.length()) {
		only1 = pref.data()[i] == '1';
		i++;
	}
	return only1;
}

bool VanitySearch::initAddress(std::string& address_str_ref, ADDRESS_ITEM* it) { 
	std::vector<unsigned char> result; // Used for Base58 decoding in BTC paths
	// `this->searchMode` is the primary mode from command line (e.g. SEARCH_COMPRESSED, ETHEREUM).
	// `this->searchType` is used internally for BTC type (P2PKH, P2SH etc.) or HASH160.
	// If `this->searchMode` is ETHEREUM, `this->searchType` will be set to ETHEREUM by this function.
	int nbDigit = 0; // Used by BTC address logic
	bool wrong = false; // Used by BTC address logic

	if (this->searchType == ETHEREUM) {
		if (address_str_ref.rfind("0x", 0) != 0) { // Ensure prefix starts with "0x"
			fprintf(stderr, "Error: Ethereum prefix \"%s\" must start with \"0x\".\n", address_str_ref.c_str());
			return false;
		}
		if (address_str_ref.length() <= 2) { // Must have characters after "0x"
			fprintf(stderr, "Error: Ethereum prefix \"%s\" is too short (must be > \"0x\").\n", address_str_ref.c_str());
			return false;
		}
		if (address_str_ref.length() > 42) { // "0x" + 40 hex characters
			fprintf(stderr, "Error: Ethereum prefix \"%s\" is too long (max 0x + 40 hex chars).\n", address_str_ref.c_str());
			return false;
		}

		std::string hex_part = address_str_ref.substr(2); // Extract hex part after "0x"
		for (char c : hex_part) {
			if (!isxdigit(c)) { // Check if all characters are valid hex digits
				fprintf(stderr, "Error: Ethereum prefix \"%s\" contains non-hex characters after \"0x\".\n", address_str_ref.c_str());
				return false;
			}
		}
		// Convert the original string (which is a reference to an element in inputAddresses) to lowercase.
		// This ensures case-insensitive matching later.
		std::transform(address_str_ref.begin(), address_str_ref.end(), address_str_ref.begin(), ::tolower);
		
		// Set ADDRESS_ITEM fields for Ethereum prefix
		it->address = (char*)address_str_ref.c_str(); // Point to the (now lowercased) string in inputAddresses
		it->addressLength = (int)address_str_ref.length();
		it->isFull = (address_str_ref.length() == 42); // Full Ethereum address is "0x" + 40 hex chars
		it->sAddress = 0; // Not used for Ethereum CPU matching logic
		it->lAddress = 0; // Not used for Ethereum CPU matching logic
		// it->hash160 is not used/set for Ethereum prefixes
		return true;
	}


	// --- Start of existing BTC/HASH160 address initialization logic ---
	// This part remains largely unchanged but needs to be guarded by `this->searchType != ETHEREUM`
	// where appropriate, or ensure `aType` is correctly determined if `this->searchType` was initially -1.

	if (address_str_ref.length() < 2) {
		fprintf(stdout, "Ignoring address \"%s\" (too short)\n", address_str_ref.c_str());
		return false;
	}

	int aType = -1; // Local variable to determine type of current address_str_ref

	// Determine aType based on address_str_ref, but only if not in ETHEREUM mode overall
	if (this->searchType == HASH160) { 
		aType = HASH160;
	} else if (this->searchType != ETHEREUM) { // Fallback to BTC type detection if global mode is not ETH
		switch (address_str_ref.data()[0]) {
		case '1': aType = P2PKH; break;
		case '3': aType = P2SH; break;
		case 'b': case 'B':
			std::string temp_addr_for_bech32 = address_str_ref;
			std::transform(temp_addr_for_bech32.begin(), temp_addr_for_bech32.end(), temp_addr_for_bech32.begin(), ::tolower);
			if (strncmp(temp_addr_for_bech32.c_str(), "bc1q", 4) == 0) {
				aType = BECH32;
				// Ensure original string in inputAddresses is also lowercased for BECH32
				std::transform(address_str_ref.begin(), address_str_ref.end(), address_str_ref.begin(), ::tolower);
			}
			break;
		}
	}


	// Validate aType against the global searchType (if already set)
	if (this->searchType != ETHEREUM) { // These checks are for non-Ethereum modes
		if (aType == -1) { // Could not determine type for a non-ETH address
			fprintf(stdout, "Ignoring address \"%s\" (must start with 1, 3, bc1q, or be valid Hash160/ETH format)\n", address_str_ref.c_str());
			return false;
		}
		if (this->searchType == -1) { // If global searchType is not yet set, set it based on first valid address
			this->searchType = aType;
		} else if (aType != this->searchType) { // Current address type mismatches global search type
			fprintf(stdout, "Ignoring address \"%s\" (Mixed address types not allowed unless ETHEREUM mode is explicitly set for all ETH prefixes)\n", address_str_ref.c_str());
			return false;
		}
	}


	// --- Process based on determined aType (for HASH160, BECH32, P2PKH, P2SH) ---
	if (aType == HASH160) {
		if (address_str_ref.length() != 40) {
			fprintf(stdout, "Warning: Hash160 \"%s\" should be 40 hex chars. GPU might require padding/truncating.\n", address_str_ref.c_str());
		}
		it->isFull = (address_str_ref.length() == 40);
		// Convert hex to binary for hash160, sAddress, lAddress (primarily for GPU)
		uint8_t hash160_bin[20];
		for (int i = 0; i < 20; i++) {
			int offset = i * 2;
			if (offset + 1 < (int)address_str_ref.length()) {
				char hex_pair[3] = { address_str_ref[offset], address_str_ref[offset + 1], 0 };
				unsigned int value;
				if (sscanf(hex_pair, "%x", &value) != 1) { /* error */ hash160_bin[i] = 0; } else { hash160_bin[i] = (uint8_t)value; }
			} else { hash160_bin[i] = 0; }
		}
		memcpy(it->hash160, hash160_bin, 20);
		it->sAddress = *(address_t*)(it->hash160);
		it->lAddress = *(addressl_t*)(it->hash160);
		it->address = (char*)address_str_ref.c_str();
		it->addressLength = (int)address_str_ref.length();
		return true;
	}
	else if (aType == BECH32) { // address_str_ref is already lowercased
		uint8_t witprog[40];
		size_t witprog_len;
		int witver;
		const char* hrp = "bc"; // or "tb" for testnet, adjust if needed

		int decode_ret = segwit_addr_decode(&witver, witprog, &witprog_len, hrp, address_str_ref.c_str());

		if (decode_ret && witprog_len == 20) { // Full Bech32 address
			it->isFull = true;
			memcpy(it->hash160, witprog, 20);
			it->sAddress = *(address_t*)(it->hash160);
			it->lAddress = *(addressl_t*)(it->hash160);
			it->address = (char*)address_str_ref.c_str();
			it->addressLength = (int)address_str_ref.length();
			return true;
		}
		// Partial Bech32 prefix
		if (address_str_ref.length() < 5) { // "bc1q" + at least one char
			fprintf(stdout, "Ignoring BECH32 address \"%s\" (too short, length<5 for prefix after bc1q)\n", address_str_ref.c_str());
			return false;
		}
		// Max length for a bech32 prefix if not full (GPU might have limits)
		// For CPU, length is less of a concern for prefix matching itself.
		// The GPU code for BECH32 uses a specific length (5 chars after "bc1q")
		// For CPU, we can be more flexible, but let's align with potential GPU limitations if any.

		uint8_t data[64]; // Buffer for decoded prefix
		memset(data, 0, 64);
		size_t decoded_data_length;
		// We need to decode the part *after* "hrp1" + separator, e.g., "bc1q" -> hrp="bc", separator='1'
		// The prefix to check is after "bc1q"
		if (!bech32_decode_nocheck(data, &decoded_data_length, address_str_ref.c_str() + 4)) {
			fprintf(stdout, "Ignoring BECH32 address \"%s\" (prefix contains invalid characters)\n", address_str_ref.c_str());
			return false;
		}
		
		it->sAddress = *(address_t*)data; // Store the first 2 bytes of the decoded prefix part
		it->isFull = false;
		it->lAddress = 0; // Not used for partial prefix matching in the same way
		it->address = (char*)address_str_ref.c_str();
		it->addressLength = (int)address_str_ref.length();
		return true;
	}
	else if (aType == P2PKH || aType == P2SH) {
		// P2PKH/P2SH
		std::string p2pkh_dummy = address_str_ref; // Use a copy for base58 manipulation
		wrong = !DecodeBase58(p2pkh_dummy, result);

		if (wrong) {
			fprintf(stdout, "Ignoring address \"%s\" (0, I, O and l not allowed)\n", address_str_ref.c_str());
			return false;
		}

		if (result.size() > 21 && result.size() <= 25) { // Full P2PKH/P2SH address (20 byte hash + 1 version + 4 checksum = 25)
			it->isFull = true;
			memcpy(it->hash160, result.data() + 1, 20);
			it->sAddress = *(address_t*)(it->hash160);
			it->lAddress = *(addressl_t*)(it->hash160);
			it->address = (char*)address_str_ref.c_str();
			it->addressLength = (int)address_str_ref.length();
			return true;
		}

		// Partial P2PKH/P2SH prefix
		if (isSingularAddress(address_str_ref)) { // Address containing only '1's (common short prefix)
			if (address_str_ref.length() > 21) { // Heuristic limit
				fprintf(stdout, "Ignoring address \"%s\" (Too many 1s for a practical prefix)\n", address_str_ref.c_str());
				return false;
			}
			it->isFull = false;
			it->sAddress = 0; // sAddress for '111...' is 0
			it->lAddress = 0;
			it->address = (char*)address_str_ref.c_str();
			it->addressLength = (int)address_str_ref.length();
			return true;
		}

		// For other partial prefixes, determine sAddress by padding with '1's
		// This logic is for GPU compatibility; CPU matching will use the string directly.
		std::string temp_for_saddr = address_str_ref;
		std::vector<unsigned char> saddr_result_bytes;
		while (saddr_result_bytes.size() < 25 && temp_for_saddr.length() < 34) { // 34 is approx max P2PKH
			DecodeBase58(temp_for_saddr, saddr_result_bytes);
			if (saddr_result_bytes.size() < 25) {
				temp_for_saddr.append("1"); // Append '1' as it's the smallest char in base58
			}
		}
		if (saddr_result_bytes.size() == 25) {
			if (this->searchType == P2SH && saddr_result_bytes[0] != 5) { // Use this->searchType
                 // This check might be too strict for prefixes if the prefix itself doesn't determine the version byte yet.
				 // For now, keep it for consistency with full address validation.
				fprintf(stdout, "Ignoring P2SH prefix \"%s\" (Padding suggests unreachable version byte)\n", address_str_ref.c_str());
				return false;
			}
			it->sAddress = *(address_t*)(saddr_result_bytes.data() + 1);
		} else {
            fprintf(stdout, "Warning: Could not determine sAddress for prefix \"%s\" by padding.\n", address_str_ref.c_str());
			it->sAddress = 0; // Default or error indicator
        }
		
		it->isFull = false;
		it->lAddress = 0;
		it->address = (char*)address_str_ref.c_str();
		it->addressLength = (int)address_str_ref.length();
		return true;
	}
	return false; // Should not be reached if logic is correct
}

void VanitySearch::enumCaseUnsentiveAddress(std::string s, std::vector<std::string>& list) {

	char letter[64];
	int letterpos[64];
	int nbLetter = 0;
	int length = (int)s.length();

	for (int i = 1; i < length; i++) {
		char c = s.data()[i];
		if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
			letter[nbLetter] = tolower(c);
			letterpos[nbLetter] = i;
			nbLetter++;
		}
	}

	int total = 1 << nbLetter;

	for (int i = 0; i < total; i++) {

		char tmp[64];
		strcpy(tmp, s.c_str());

		for (int j = 0; j < nbLetter; j++) {
			int mask = 1 << j;
			if (mask & i) tmp[letterpos[j]] = toupper(letter[j]);
			else         tmp[letterpos[j]] = letter[j];
		}

		list.push_back(string(tmp));

	}

}

// ----------------------------------------------------------------------------

std::string VanitySearch::hash160ToHex(uint8_t* hash160) {
	char tmp[41];
	for (int j = 0; j < 20; j++) {
		sprintf(tmp + (j * 2), "%02X", hash160[j]);
	}
	tmp[40] = 0;
	return std::string(tmp);
}

void VanitySearch::output(string addr, string pAddr, string pAddrHex, std::string pubKey) {

#ifdef WIN64
	WaitForSingleObject(ghMutex, INFINITE);
#else
	pthread_mutex_lock(&ghMutex);
#endif

	FILE* f = stdout;
	bool needToClose = false;

	if (outputFile.length() > 0) {
		f = fopen(outputFile.c_str(), "a");
		if (f == NULL) {
			fprintf(stderr, "Cannot open %s for writing\n", outputFile.c_str());
			f = stdout;
		}
		else {
			needToClose = true;
		}
	}

	if (searchType == HASH160) {
		// For Hash160 search, show both the Hash160 and the corresponding Bitcoin address
		// Extract the "raw" hash160 from the addr string to display it in canonical form
		string hash160Hex = addr.substr(0, 40); // First 40 chars should be the hex representation
		
		// Get the corresponding Bitcoin address (always use compressed for better usability)
		Int privKey;
		char* hexCopy = new char[pAddrHex.length() + 1];
		strcpy(hexCopy, pAddrHex.c_str());
		privKey.SetBase16(hexCopy);
		delete[] hexCopy;
		Point p = secp->ComputePublicKey(&privKey);
		string btcAddr = secp->GetAddress(P2PKH, true, p);
		
		if (f != stdout)
			fprintf(f, "\nHash160: %s\n", hash160Hex.c_str());
		fprintf(stdout, "\nHash160: %s\n", hash160Hex.c_str());
		
		if (f != stdout)
			fprintf(f, "Bitcoin Address: %s\n", btcAddr.c_str());
		fprintf(stdout, "Bitcoin Address: %s\n", btcAddr.c_str());
	} else {
		// Normal address output
		if (f != stdout)
			fprintf(f, "\nPublic Addr: %s\n", addr.c_str());	
		fprintf(stdout, "\nPublic Addr: %s\n", addr.c_str());
	}

	switch (searchType) {
	case P2PKH:
		if (f != stdout)
			fprintf(f, "Priv (WIF): p2pkh:%s\n", pAddr.c_str());
		fprintf(stdout, "Priv (WIF): p2pkh:%s\n", pAddr.c_str());
		break;
	case P2SH:
		if (f != stdout)
			fprintf(f, "Priv (WIF): p2wpkh-p2sh:%s\n", pAddr.c_str());
		fprintf(stdout, "Priv (WIF): p2wpkh-p2sh:%s\n", pAddr.c_str());
		break;
	case BECH32:
		if (f != stdout)
			fprintf(f, "Priv (WIF): p2wpkh:%s\n", pAddr.c_str());
		fprintf(stdout, "Priv (WIF): p2wpkh:%s\n", pAddr.c_str());
		break;
	case HASH160:
		if (f != stdout)
			fprintf(f, "Priv (WIF): %s\n", pAddr.c_str());
		fprintf(stdout, "Priv (WIF): %s\n", pAddr.c_str());
		break;
	case ETHEREUM: // New case for Ethereum
		if (f != stdout) {
			fprintf(f, "Priv (HEX): 0x%s\n", pAddr.c_str()); // pAddr is used as privKeyRepresentation (Ethereum hex private key)
			// Optionally print public key:
			// if (!pubKey.empty()) {
			//     fprintf(f, "Pub Key (HEX): %s\n", pubKey.c_str());
			// }
		}
		fprintf(stdout, "Priv (HEX): 0x%s\n", pAddr.c_str()); // pAddr is used as privKeyRepresentation
		// Optionally print public key to stdout:
		// if (!pubKey.empty()) {
		//     fprintf(stdout, "Pub Key (HEX): %s\n", pubKey.c_str());
		// }
		break;
	}

	// For non-Ethereum types, pAddrHex is the second private key representation.
	// For Ethereum, this line is effectively skipped due to the break in the ETHEREUM case,
	// or pAddrHex would be empty if not specifically populated for ETH.
	// However, to be clean, we can conditionalize this.
	if (searchType != ETHEREUM) {
		if (f != stdout)
			fprintf(f, "Priv (HEX): 0x%s\n", pAddrHex.c_str());	
		fprintf(stdout, "Priv (HEX): 0x%s\n", pAddrHex.c_str());
	}
	fprintf(stdout, "\n");

	if (f != stdout)
		fflush(f);
	fflush(stdout);
	//fflush(stderr);	

	if (needToClose)
		fclose(f);

#ifdef WIN64
	ReleaseMutex(ghMutex);
#else
	pthread_mutex_unlock(&ghMutex);
#endif
}

void VanitySearch::updateFound() {

	// Check if all addresses has been found
	// Needed only if stopWhenFound is asked
	if (stopWhenFound) 	{

		bool allFound = true;
		for (int i = 0; i < (int)usedAddress.size(); i++) {
			bool iFound = true;
			address_t p = usedAddress[i];
			if (!addresses[p].found) {
				if (addresses[p].items) {
					for (int j = 0; j < (int)addresses[p].items->size(); j++) {
						iFound &= *((*addresses[p].items)[j].found);
					}
				}
				addresses[usedAddress[i]].found = iFound;
			}
			allFound &= iFound;
		}

		endOfSearch = allFound;		
	}		
}

bool VanitySearch::checkPrivKey(string addr_matched, Int& key_base_priv, int32_t incr_priv, int endomorphism_priv, bool mode_comp_priv) {

	if (this->searchType == ETHEREUM) {
		Int k_actual(key_base_priv);
		if (incr_priv < 0) {
			k_actual.Add((uint64_t)(-incr_priv));
			k_actual.Neg();
			k_actual.Add(&secp->order);
		} else {
			k_actual.Add((uint64_t)incr_priv);
		}

		// Apply endomorphism (assuming endomorphism_priv = 0 for typical Ethereum search for now)
		// This part might need to be conditional if endomorphism isn't used for ETH.
		// For now, include it for completeness, matching potential structure.
		switch (endomorphism_priv) {
			case 1: k_actual.ModMulK1order(&lambda); break; // lambda is a class member
			case 2: k_actual.ModMulK1order(&lambda2); break; // lambda2 is a class member
		}

		Point p = secp->ComputePublicKey(&k_actual);
		std::string chkAddr = secp->GetEthereumAddress(p); // Returns lowercase "0x..."

		// addr_matched is assumed to be lowercase "0x..." from checkAddr
		if (chkAddr == addr_matched) {
			// Parameters for output:
			// 1. Matched address string (addr_matched)
			// 2. Ethereum private key as HEX string (from secp->GetEthereumPrivateKeyHex(&k_actual))
			// 3. Bitcoin WIF private key hex (k_actual.GetBase16()) - can still be useful as raw hex, but output fn will ignore for ETH.
			// 4. Public key hex string (from secp->GetPublicKeyHex(false, p)) - optional
			output(addr_matched, 
				   secp->GetEthereumPrivateKeyHex(k_actual), // This is the primary private key string for ETH output
				   k_actual.GetBase16(),                     // Raw private key hex, passed as the 3rd param. Output fn will ignore for ETH.
				   secp->GetPublicKeyHex(false, p));         // Full uncompressed public key
			return true;
		} else {
			fprintf(stdout, "\nWarning, Ethereum address mismatch in checkPrivKey!\n");
			fprintf(stdout, "  Expected: %s\n", addr_matched.c_str());
			fprintf(stdout, "  Derived:  %s\n", chkAddr.c_str());
			return false;
		}
	}
	// Else, the existing Bitcoin logic follows
	else {
		Int k(key_base_priv);	

		if (incr_priv < 0) {
			k.Add((uint64_t)(-incr_priv));
			k.Neg();
			k.Add(&secp->order);		
		}
		else {
			k.Add((uint64_t)incr_priv);
		}

		// Endomorphisms
		switch (endomorphism_priv) {
		case 1:
			k.ModMulK1order(&lambda);		
			break;
		case 2:
			k.ModMulK1order(&lambda2);		
			break;
		}

		// Check addresses
		Point p = secp->ComputePublicKey(&k);	

		string chkAddr = secp->GetAddress(searchType, mode_comp_priv, p);
		if (chkAddr != addr_matched) {

			// Key may be the opposite one (negative zero or compressed key)
			k.Neg();
			k.Add(&secp->order);
			p = secp->ComputePublicKey(&k);
			
			string chkAddr_neg = secp->GetAddress(searchType, mode_comp_priv, p); // Renamed to avoid conflict
			if (chkAddr_neg != addr_matched) {
				fprintf(stdout, "\nWarning, wrong private key generated !\n");
				fprintf(stdout, "  Addr :%s\n", addr_matched.c_str());
				fprintf(stdout, "  Check:%s\n", chkAddr.c_str()); // Original chkAddr
				fprintf(stdout, "  Check (neg):%s\n", chkAddr_neg.c_str()); // Check with negated key
				fprintf(stdout, "  Endo:%d incr:%d comp:%d\n", endomorphism_priv, incr_priv, mode_comp_priv);
				return false;
			}
		}

		output(addr_matched, secp->GetPrivAddress(mode_comp_priv, k), k.GetBase16(), secp->GetPublicKeyHex(mode_comp_priv, p));

		return true;
	}
}

void VanitySearch::checkAddrSSE(uint8_t* h1, uint8_t* h2, uint8_t* h3, uint8_t* h4,
	int32_t incr1, int32_t incr2, int32_t incr3, int32_t incr4,
	Int& key, int endomorphism, bool mode) {

	vector<string> addr = secp->GetAddress(searchType, mode, h1, h2, h3, h4);

	for (int i = 0; i < (int)inputAddresses.size(); i++) {

		if (Wildcard::match(addr[0].c_str(), inputAddresses[i].c_str())) {

			// Found it !      
			if (checkPrivKey(addr[0], key, incr1, endomorphism, mode)) {
				nbFoundKey++;
				//patternFound[i] = true;
				updateFound();
			}
		}

		if (Wildcard::match(addr[1].c_str(), inputAddresses[i].c_str())) {

			// Found it !      
			if (checkPrivKey(addr[1], key, incr2, endomorphism, mode)) {
				nbFoundKey++;
				//patternFound[i] = true;
				updateFound();
			}
		}

		if (Wildcard::match(addr[2].c_str(), inputAddresses[i].c_str())) {

			// Found it !      
			if (checkPrivKey(addr[2], key, incr3, endomorphism, mode)) {
				nbFoundKey++;
				//patternFound[i] = true;
				updateFound();
			}
		}

		if (Wildcard::match(addr[3].c_str(), inputAddresses[i].c_str())) {

			// Found it !      
			if (checkPrivKey(addr[3], key, incr4, endomorphism, mode)) {
				nbFoundKey++;
				//patternFound[i] = true;
				updateFound();
			}
		}
	}
}

void VanitySearch::checkAddr(int prefIdx, uint8_t* hash160_btc, Int& key_base, int32_t incr, int endomorphism, bool mode_bitcoin) {
	
	if (this->searchType == ETHEREUM) {
		Int k_actual(key_base); // Use key_base from parameters
		if (incr < 0) {
			k_actual.Add((uint64_t)(-incr));
			k_actual.Neg();
			k_actual.Add(&secp->order);
		} else {
			k_actual.Add((uint64_t)incr);
		}

		switch (endomorphism) {
			case 1: k_actual.ModMulK1order(&lambda); break;
			case 2: k_actual.ModMulK1order(&lambda2); break;
		}

		Point p = secp->ComputePublicKey(&k_actual);
		std::string ethAddrFull = secp->GetEthereumAddress(p); // Lowercase "0x..."

		// Iterate through all target prefixes stored in this->inputAddresses
		for (size_t i = 0; i < this->inputAddresses.size(); ++i) {
			// this->inputAddresses[i] was already lowercased and validated by initAddress
			const std::string& targetPrefixString = this->inputAddresses[i]; 

			if (ethAddrFull.rfind(targetPrefixString, 0) == 0) { // string::rfind for prefix check
				// Match found! Call checkPrivKey to verify and output.
				// mode_bitcoin (compression) is irrelevant for ETH path in checkPrivKey.
				// Parameters to checkPrivKey: matched_address, base_key, increment, endomorphism_type, compression_mode (false for ETH)
				if (checkPrivKey(ethAddrFull, key_base, incr, endomorphism, false)) { 
					// nbFoundKey and updateFound() are called within checkPrivKey -> output -> updateFound path.
					// updateFound() handles setting this->endOfSearch if stopWhenFound is true and all are found.
					if (this->stopWhenFound && (this->nbFoundKey >= this->inputAddresses.size() || this->nbFoundKey >= this->maxFound) ) { 
						 this->endOfSearch = true; // Signal to stop searching in other threads/contexts
						 return; // Exit checkAddr early
					}
				}
			}
		}
		return; // Finished checking all Ethereum prefixes for this key
	}
	// Else, the existing Bitcoin logic follows (which uses prefIdx and hash160_btc)
	else {
		vector<ADDRESS_ITEM>* pi = addresses[prefIdx].items;	

		if (onlyFull) {
			// Full addresses
			for (int i = 0; i < (int)pi->size(); i++) {

				if (stopWhenFound && *((*pi)[i].found))
					continue;

				bool match = false;
			
				if (this->searchType == HASH160) { // Use this->searchType
					match = memcmp((*pi)[i].hash160, hash160_btc, 20) == 0;
				} else {
					match = ripemd160_comp_hash((*pi)[i].hash160, hash160_btc);
				}

				if (match) {
					// Found it!
					*((*pi)[i].found) = true;
					// You believe it?
				
					string address;
					if (this->searchType == HASH160) { // Use this->searchType
						address = hash160ToHex(hash160_btc);
					} else {
						address = secp->GetAddress(this->searchType, mode_bitcoin, hash160_btc); // Use this->searchType
					}
				
					if (checkPrivKey(address, key_base, incr, endomorphism, mode_bitcoin)) {
						// nbFoundKey is incremented in output -> checkPrivKey path
						updateFound(); 
					}
				}
			}
		}
		else { // Not onlyFull (partial match logic for BTC types)
			char a[64]; // Buffer for prefix comparison

			string addr_str_from_hash; 
			if (this->searchType == HASH160) { // Use this->searchType
				addr_str_from_hash = hash160ToHex(hash160_btc);
			} else {
				addr_str_from_hash = secp->GetAddress(this->searchType, mode_bitcoin, hash160_btc); // Use this->searchType
			}

			for (int i = 0; i < (int)pi->size(); i++) {
				if (stopWhenFound && *((*pi)[i].found))
					continue;

				if (this->searchType == HASH160) { // Use this->searchType
					string prefix_to_match((*pi)[i].address, (*pi)[i].addressLength);
					if (addr_str_from_hash.rfind(prefix_to_match, 0) == 0) { 
						*((*pi)[i].found) = true;
						if (checkPrivKey(addr_str_from_hash, key_base, incr, endomorphism, mode_bitcoin)) {
							updateFound();
						}
					}
				} else {
					if ((*pi)[i].addressLength < 64) { // Ensure no buffer overflow
						strncpy(a, addr_str_from_hash.c_str(), (*pi)[i].addressLength);
						a[(*pi)[i].addressLength] = 0;

						if (strcmp((*pi)[i].address, a) == 0) {
							*((*pi)[i].found) = true;
							if (checkPrivKey(addr_str_from_hash, key_base, incr, endomorphism, mode_bitcoin)) {
								updateFound();
							}
						}
					}
				}
			}
		}
	} // End of existing Bitcoin logic (else block)
}

#ifdef WIN64
DWORD WINAPI _FindKeyGPU(LPVOID lpParam) {
#else
void* _FindKeyGPU(void* lpParam) {
#endif
	TH_PARAM* p = (TH_PARAM*)lpParam;
	p->obj->FindKeyGPU(p);
	return 0;
}

void VanitySearch::checkAddresses(bool compressed, Int key, int i, Point p1) {

	unsigned char h0[20];
	Point pte1[1];
	Point pte2[1];

	// Point
	secp->GetHash160(searchType, compressed, p1, h0);
	address_t pr0 = *(address_t*)h0;
	if (addresses[pr0].items)
		checkAddr(pr0, h0, key, i, 0, compressed);	
}

void VanitySearch::checkAddressesSSE(bool compressed, Int key, int i, Point p1, Point p2, Point p3, Point p4) {

	unsigned char h0[20];
	unsigned char h1[20];
	unsigned char h2[20];
	unsigned char h3[20];
	Point pte1[4];
	Point pte2[4];
	address_t pr0;
	address_t pr1;
	address_t pr2;
	address_t pr3;

	// Point -------------------------------------------------------------------------
	secp->GetHash160(searchType, compressed, p1, p2, p3, p4, h0, h1, h2, h3);	

	pr0 = *(address_t*)h0;
	pr1 = *(address_t*)h1;
	pr2 = *(address_t*)h2;
	pr3 = *(address_t*)h3;

	if (addresses[pr0].items)
		checkAddr(pr0, h0, key, i, 0, compressed);
	if (addresses[pr1].items)
		checkAddr(pr1, h1, key, i + 1, 0, compressed);
	if (addresses[pr2].items)
		checkAddr(pr2, h2, key, i + 2, 0, compressed);
	if (addresses[pr3].items)
		checkAddr(pr3, h3, key, i + 3, 0, compressed);	
}

void VanitySearch::getGPUStartingKeys(Int& tRangeStart, Int& tRangeEnd, int groupSize, int nbThread, Point *p, uint64_t Progress) {
		
	int grp_startkeys = nbThread/256;

	//New setting key by fixedpaul using addition on secp with batch modular inverse, super fast, multithreading not needed

	Int stepThread;
	Int numthread;

	stepThread.Set(&bc->ksFinish);
	stepThread.Sub(&bc->ksStart);
	stepThread.AddOne();
	numthread.SetInt32(nbThread);
	stepThread.Div(&numthread);

	Point Pdouble;
	Int kDouble;

	kDouble.Set(&stepThread);
	kDouble.Mult(grp_startkeys);
	Pdouble = secp->ComputePublicKey(&kDouble);

	Point P_start;
	Int kStart;

	kStart.Set(&stepThread);
	kStart.Mult(grp_startkeys / 2);
	kStart.Add(groupSize / 2 + Progress);

	

	P_start = secp->ComputePublicKey(&kStart);

	p[grp_startkeys / 2] = secp->ComputePublicKey(&tRangeStart);
	p[grp_startkeys / 2] = secp->AddDirect(p[grp_startkeys / 2], P_start);


	Int key_delta;
	Point* p_delta;
	p_delta = new Point[grp_startkeys / 2];

	key_delta.Set(&stepThread);

	

	p_delta[0] = secp->ComputePublicKey(&key_delta);
	key_delta.Add(&stepThread);
	p_delta[1] = secp->ComputePublicKey(&key_delta);

	for (int i = 2; i < grp_startkeys / 2; i++) {
		p_delta[i] = secp->AddDirect(p_delta[i - 1], p_delta[0]);
	}

	Int* dx;
	Int* subp;

	subp = new Int[grp_startkeys / 2 + 1];
	dx = new Int[grp_startkeys / 2 + 1];

	uint32_t j;
	uint32_t i;

	for (i = grp_startkeys / 2; i < nbThread; i += grp_startkeys) {

		double percentage = (100.0 * (double)(i + grp_startkeys / 2)) / (double)(nbThread);
		printf("Setting starting keys... [%.2f%%] \r", percentage);
		fflush(stdout);


		for (j = 0; j < grp_startkeys / 2; j++) {
			dx[j].ModSub(&p_delta[j].x, &p[i].x);
		}
		dx[grp_startkeys / 2].ModSub(&Pdouble.x, &p[i].x);

		Int newValue;
		Int inverse;

		subp[0].Set(&dx[0]);
		for (int j = 1; j < grp_startkeys / 2 + 1; j++) {
			subp[j].ModMulK1(&subp[j - 1], &dx[j]);
		}

		// Do the inversion - using batch modular inverse
		inverse.Set(&subp[grp_startkeys / 2]);
		inverse.ModInv();

		for (j = grp_startkeys / 2; j > 0; j--) {
			newValue.ModMulK1(&subp[j - 1], &inverse);
			inverse.ModMulK1(&dx[j]);
			dx[j].Set(&newValue);
		}

		dx[0].Set(&inverse);

		Int _s;
		Int _p;
		Int dy;
		Int syn;
		syn.Set(&p[i].y);
		syn.ModNeg();



		for (j = 0; j < grp_startkeys / 2 - 1; j++) {

			dy.ModSub(&p_delta[j].y, &p[i].y);
			_s.ModMulK1(&dy, &dx[j]);

			_p.ModSquareK1(&_s);

			p[i + j + 1].x.ModSub(&_p, &p[i].x);
			p[i + j + 1].x.ModSub(&p_delta[j].x);

			p[i + j + 1].y.ModSub(&p_delta[j].x, &p[i + j + 1].x);
			p[i + j + 1].y.ModMulK1(&_s);
			p[i + j + 1].y.ModSub(&p_delta[j].y);

			dy.ModSub(&syn, &p_delta[j].y);
			_s.ModMulK1(&dy, &dx[j]);

			_p.ModSquareK1(&_s);

			p[i - j - 1].x.ModSub(&_p, &p[i].x);
			p[i - j - 1].x.ModSub(&p_delta[j].x);

			p[i - j - 1].y.ModSub(&p[i - j - 1].x, &p_delta[j].x);
			p[i - j - 1].y.ModMulK1(&_s);
			p[i - j - 1].y.ModSub(&p_delta[j].y, &p[i - j - 1].y);
		}

		dy.ModSub(&syn, &p_delta[j].y);
		_s.ModMulK1(&dy, &dx[j]);

		_p.ModSquareK1(&_s);


		p[i - j - 1].x.ModSub(&_p, &p[i].x);
		p[i - j - 1].x.ModSub(&p_delta[j].x);

		p[i - j - 1].y.ModSub(&p[i - j - 1].x, &p_delta[j].x);
		p[i - j - 1].y.ModMulK1(&_s);
		p[i - j - 1].y.ModSub(&p_delta[j].y, &p[i - j - 1].y);

		if (i + grp_startkeys < nbThread) {

			dy.ModSub(&Pdouble.y, &p[i].y);
			_s.ModMulK1(&dy, &dx[grp_startkeys / 2]);

			_p.ModSquareK1(&_s);

			p[i + grp_startkeys].x.ModSub(&_p, &p[i].x);
			p[i + grp_startkeys].x.ModSub(&Pdouble.x);

			p[i + grp_startkeys].y.ModSub(&Pdouble.x, &p[i + grp_startkeys].x);
			p[i + grp_startkeys].y.ModMulK1(&_s);
			p[i + grp_startkeys].y.ModSub(&Pdouble.y);
		}
	}

	delete[] subp;
	delete[] dx;
	delete[] p_delta;
}

void VanitySearch::FindKeyGPU(TH_PARAM* ph) {

	bool ok = true;

	double t0;
	double ttot;
	uint64_t keys_n = 0;
	static uint64_t keys_n_prev = 0;
	static double tprev = 0.0;

	// Global init
	int thId = ph->threadId;
	GPUEngine g(ph->gpuId, maxFound);
	int numThreadsGPU = g.GetNbThread();
	int STEP_SIZE = g.GetStepSize();
	Point* publicKeys = new Point[numThreadsGPU];
	vector<ITEM> found;

	Point RandomJump_P;
	Int RandomJump_K;
	Int RandomJump_K_last;
	Int RandomJump_K_tot;
	RandomJump_K.SetInt32(STEP_SIZE);
	RandomJump_K_last.SetInt32(0);
	RandomJump_K_tot.SetInt32(0);

	// Create a different key range for each GPU
	Int keySpaceSize;
	Int gpuKeySpaceSize;
	Int startPosition;

	keySpaceSize.Set(&bc->ksFinish);
	keySpaceSize.Sub(&bc->ksStart);
	keySpaceSize.AddOne();

	gpuKeySpaceSize.Set(&keySpaceSize);
	Int gpuCount;
	gpuCount.SetInt32(numGPUs);
	gpuKeySpaceSize.Div(&gpuCount);

	startPosition.Set(&gpuKeySpaceSize);
	startPosition.Mult(thId);
	startPosition.Add(&bc->ksStart);

	// Create ending position for this GPU
	Int endPosition;
	endPosition.Set(&startPosition);
	endPosition.Add(&gpuKeySpaceSize);
	if (thId == numGPUs - 1) {
		// Make sure the last GPU covers the remainder of the range
		endPosition.Set(&bc->ksFinish);
		endPosition.AddOne();
	}

	fprintf(stdout, "GPU #%d: %s\n", ph->gpuId, g.deviceName.c_str());
	fprintf(stdout, "GPU #%d: Search range: 0x%s -> 0x%s\n", 
		ph->gpuId, 
		startPosition.GetBase16().c_str(),
		endPosition.GetBase16().c_str());
	fflush(stdout);

	counters[thId] = 0;
	
	g.SetSearchMode(searchMode);
	g.SetSearchType(searchType);
	if (onlyFull) {
		g.SetAddress(usedAddressL, nbAddress);
	}
	else {
		g.SetAddress(usedAddress);
	}

	Int stepThread;
	Int taskSize;
	Int numthread;

	taskSize.Set(&endPosition);
	taskSize.Sub(&startPosition);
	numthread.SetInt32(numThreadsGPU);
	stepThread.Set(&taskSize);
	stepThread.Div(&numthread);

	Int privkey;
	Int part_key;
	Int keycount;

	t0 = Timer::get_tick();

	getGPUStartingKeys(startPosition, endPosition, g.GetGroupSize(), numThreadsGPU, publicKeys, (uint64_t)(1ULL * idxcount * g.GetStepSize()));

	ok = g.SetKeys(publicKeys);
	delete[] publicKeys;

	ttot = Timer::get_tick() - t0;

	printf("GPU #%d: Starting keys set in %.2f seconds \n", ph->gpuId, ttot);
	fflush(stdout);

	ph->hasStarted = true;

	printf("GPU #%d: Started! \r", ph->gpuId);
	fflush(stdout);

	t0 = Timer::get_tick();

	endOfSearch = false;

	while (ok && !endOfSearch) {

		if (!Pause) {	
			if (randomMode) {
				RandomJump_K_last.Set(&RandomJump_K);
				RandomJump_K_tot.Add(&RandomJump_K);

				RandomJump_K.Rand(256);
				RandomJump_K.Mod(&stepThread);
				RandomJump_K.Sub(&RandomJump_K_tot);
				
				if (RandomJump_K.IsNegative()) {
					RandomJump_K.Neg();
					RandomJump_P = secp->ComputePublicKey(&RandomJump_K);
					RandomJump_P.y.ModNeg();
					RandomJump_K.Neg();
				}
				else {
					RandomJump_P = secp->ComputePublicKey(&RandomJump_K);
				}
				
				ok = g.SetRandomJump(RandomJump_P);
			}

			ok = g.Launch(found, true);
			idxcount += 1;

			ttot = Timer::get_tick() - t0 + t_Paused;

			keycount.SetInt32(idxcount - 1);
			keycount.Mult(STEP_SIZE);


			for (int i = 0; i < (int)found.size() && !endOfSearch; i++) {

				ITEM it = found[i];
				part_key.Set(&stepThread);
				part_key.Mult(it.thId);
	
				privkey.Set(&startPosition);
				privkey.Add(&part_key);

				if (randomMode) {
					privkey.Add(&RandomJump_K_tot);
					privkey.Sub(&RandomJump_K_last);
				}
				else {				
					privkey.Add(&keycount);
				}
	
				checkAddr(*(address_t*)(it.hash), it.hash, privkey, it.incr, it.endo, it.mode);
			}

			keycount.Add(STEP_SIZE);
			keycount.Mult(numThreadsGPU);

			keys_n = 1ULL * STEP_SIZE * numThreadsGPU;
			keys_n = keys_n * idxcount;

		} else {
			printf("GPU #%d: Pausing...\r", ph->gpuId);
			fflush(stdout);

			g.FreeGPUEngine();

			Paused = true;
			t_Paused = ttot;
		}
		
		PrintStats(keys_n, keys_n_prev, ttot, tprev, taskSize, keycount);

		if (!randomMode && (keycount.IsGreaterOrEqual(&taskSize)))
		{
			double avg_speed = static_cast<double>(keys_n) / (ttot * 1000000.0); // Avg speed in MK/s
			printf("\n");
			printf("GPU #%d: Range Finished! - Average Speed: %.1f [MK/s] - Found: %d   \r", 
				ph->gpuId, avg_speed, nbFoundKey);
			printf("\n");
			fflush(stdout);

			char* ctimeBuff;
			time_t now = time(NULL);
			ctimeBuff = ctime(&now);
			printf("GPU #%d: Current task END time: %s", ph->gpuId, ctimeBuff);

			// This GPU is done, but we don't set endOfSearch here
			// to allow other GPUs to continue running
			ph->isRunning = false;
			break;
		}

		keys_n_prev = keys_n;
		tprev = ttot;
	}

	ph->isRunning = false;

	// We set endOfSearch when all GPUs are done or when stop is requested
	if (nbFoundKey > 0 && stopWhenFound) {
		endOfSearch = true;
	}
}


void VanitySearch::PrintStats(uint64_t keys_n, uint64_t keys_n_prev, double ttot, double tprev, Int taskSize, Int keycount) {

	// Try to get the GPU ID of the calling thread 
	int gpuId = -1;
	if (threads != nullptr) {
#ifdef WIN64
		DWORD currentThreadId = GetCurrentThreadId();
		for (int i = 0; i < numGPUs; i++) {
			if (GetThreadId(threads[i].native_handle()) == currentThreadId) {
				gpuId = i;
				break;
			}
		}
#else
		pthread_t currentThread = pthread_self();
		for (int i = 0; i < numGPUs; i++) {
			if (pthread_equal(currentThread, threads[i].native_handle())) {
				gpuId = i;
				break;
			}
		}
#endif
	}

	// Continue with the rest of the method
	double speed;
	double perc;
	double log_keys;

	Int Perc;

	Perc.Set(&taskSize);
	Perc.Mult(65536);
	Perc.Div(&keycount);


	if (ttot > tprev) {
		speed = (keys_n - keys_n_prev) / (ttot - tprev) / 1000000.0; // speed in Mkey/s
	}


	perc = (double)(1 / Perc.ToDouble()*100*65536);


	log_keys = log2(static_cast<double>(keys_n));

	int h_run = static_cast<int32_t>(ttot) / 3600;
	int m_run = (static_cast<int32_t>(ttot) % 3600) / 60;
	int s_run = static_cast<int32_t>(ttot) % 60;
	int d_run = static_cast<int32_t>(ttot * 10) % 10;

	double tempo_tot_stimato = ttot / (perc / 100.0);
	double end_tt = tempo_tot_stimato - ttot;

	int h_end = static_cast<int32_t>(end_tt) / 3600;
	int m_end = (static_cast<int32_t>(end_tt) % 3600) / 60;
	int s_end = static_cast<int32_t>(end_tt) % 60;
	int d_end = static_cast<int32_t>(end_tt * 10) % 10;

	std::string gpuPrefix = (gpuId >= 0) ? "GPU #" + std::to_string(gpuId) + ": " : "";

	if (randomMode) {
		if (!Paused) {

			printf("%s%.1f MK/s - 2^%.2f [%.2f%%] - RUN: %02d:%02d:%02d.%01d - Found: %d     ",
				gpuPrefix.c_str(), speed, log_keys, perc, h_run, m_run, s_run, d_run, nbFoundKey);

		}
		else {
			printf("%sPaused - 2^%.2f [%.2f%%] - RUN: %02d:%02d:%02d.%01d - Found: %d     ",
				gpuPrefix.c_str(), log_keys, perc, h_run, m_run, s_run, d_run, nbFoundKey);

			endOfSearch = true;
		}
	}
	else {
		if (!Paused) {

			if (h_end >= 0)
				printf("%s%.1f MK/s - 2^%.2f [%.2f%%] - RUN: %02d:%02d:%02d.%01d|END: %02d:%02d:%02d.%01d - Found: %d     ",
					gpuPrefix.c_str(), speed, log_keys, perc, h_run, m_run, s_run, d_run, h_end, m_end, s_end, d_end, nbFoundKey);
			else
				printf("%s%.1f MK/s - 2^%.2f [%.2f%%] - RUN: %02d:%02d:%02d.%01d|END: Too much bro - Found: %d     ",
					gpuPrefix.c_str(), speed, log_keys, perc, h_run, m_run, s_run, d_run, nbFoundKey);
		}
		else {
			printf("%sPaused - 2^%.2f [%.2f%%] - RUN: %02d:%02d:%02d.%01d|END: %02d:%02d:%02d.%01d - Found: %d     ",
				gpuPrefix.c_str(), log_keys, perc, h_run, m_run, s_run, d_run, h_end, m_end, s_end, d_end, nbFoundKey);

			endOfSearch = true;
		}
	}


	printf("\r");


	fflush(stdout);
}

bool VanitySearch::isAlive(TH_PARAM * p) {

	bool isAlive = true;
	int total = numGPUs;
	for (int i = 0; i < total; i++)
		isAlive = isAlive && p[i].isRunning;

	return isAlive;
}

bool VanitySearch::hasStarted(TH_PARAM * p) {

	bool hasStarted = true;
	int total = numGPUs;
	for (int i = 0; i < total; i++)
		hasStarted = hasStarted && p[i].hasStarted;

	return hasStarted;
}

uint64_t VanitySearch::getGPUCount() {

	uint64_t count = 0;
	for (int i = 0; i < numGPUs; i++) {
		count += counters[i];
	}
	return count;
}

void VanitySearch::saveProgress(TH_PARAM* p, Int& lastSaveKey, BITCRACK_PARAM* bc) {

	Int lowerKey;
	lowerKey.Set(&p[0].THnextKey);

	int total = numGPUs;
	for (int i = 0; i < total; i++) {
		if (p[i].THnextKey.IsLower(&lowerKey))
			lowerKey.Set(&p[i].THnextKey);
	}

	if (lowerKey.IsLowerOrEqual(&lastSaveKey)) return;
	lastSaveKey.Set(&lowerKey);
}

void VanitySearch::Search(std::vector<int> gpuId, std::vector<int> gridSize) {

	double t0;
	double t1;
	endOfSearch = false;
	numGPUs = (int)gpuId.size();
	nbFoundKey = 0;

	memset(counters, 0, sizeof(counters));	

	TH_PARAM* params = (TH_PARAM*)malloc(numGPUs * sizeof(TH_PARAM));
	memset(params, 0, numGPUs * sizeof(TH_PARAM));
	
	threads = new std::thread[numGPUs];

#ifdef WIN64
	ghMutex = CreateMutex(NULL, FALSE, NULL);
	mutex = CreateMutex(NULL, FALSE, NULL);
#else
	ghMutex = PTHREAD_MUTEX_INITIALIZER;
	mutex = PTHREAD_MUTEX_INITIALIZER;
#endif

	// Launch GPU threads
	for (int i = 0; i < numGPUs; i++) {
		params[i].obj = this;
		params[i].threadId = i;
		params[i].isRunning = true;
		params[i].gpuId = gpuId[i];
		params[i].gridSizeX = gridSize[i * 2];
		params[i].gridSizeY = gridSize[i * 2 + 1];
		params[i].THnextKey.Set(&bc->ksNext);
		params[i].hasStarted = false;
		
		threads[i] = std::thread(_FindKeyGPU, params + i);
	}

	// Wait for all threads to start
	while (!hasStarted(params)) {
		Timer::SleepMillis(500);
	}

	// Wait for end of search or Ctrl+C
	while (!endOfSearch) {
		Timer::SleepMillis(100);
	}

	// Wait for threads to finish
	for (int i = 0; i < numGPUs; i++) {
		if(threads[i].joinable())
			threads[i].join();
	}

	delete[] threads;
	threads = nullptr;
	
	if (params != nullptr) {
		free(params);
	}
}

string VanitySearch::GetHex(vector<unsigned char> &buffer) {

	string ret;

	char tmp[128];
	for (int i = 0; i < (int)buffer.size(); i++) {
		sprintf(tmp, "%02hhX", buffer[i]);
		ret.append(tmp);
	}

	return ret;
}
