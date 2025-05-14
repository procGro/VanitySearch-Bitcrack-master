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


#include <sstream> 
#include "Timer.h"
#include "Vanity.h"
#include "SECP256k1.h"
#include <fstream>
#include <string>
#include <string.h>
#include <stdexcept>
#include "hash/sha512.h"
#include "hash/sha256.h"
#include <thread>
#include <atomic>
#include <algorithm>  // For std::transform
#include "Base58.h"   // For DecodeBase58
#include "Bech32.h"   // For segwit_addr_decode

#if defined(_WIN32) || defined(_WIN64)
#include <conio.h> // For _kbhit and _getch on Windows
#else
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#endif

std::atomic<bool> Pause(false);
std::atomic<bool> Paused(false);
std::atomic<bool> stopMonitorKey(false);
int idxcount;
double t_Paused;
bool randomMode = false;

// Forward declaration of function defined at the end of this file
std::vector<int> parseGPUIds(const std::string& gpuParsed);

#if defined(_WIN32) || defined(_WIN64)
void monitorKeypress() {
	while (!stopMonitorKey) {
		Timer::SleepMillis(1);
		if (_kbhit()) { // Check if a key is pressed
			char ch = _getch(); // Get the pressed key
			if (ch == 'p' || ch == 'P') {
				Pause = !(Pause);
				//printf("\nPause PRESSED\n");
				//break;
			}
		}
	}
}
#else
void setTerminalRawMode(bool enable) {
	static struct termios oldt, newt;
	if (enable) {
		tcgetattr(STDIN_FILENO, &oldt); // Get current terminal settings
		newt = oldt;
		newt.c_lflag &= ~(ICANON | ECHO); // Disable canonical mode and echo
		tcsetattr(STDIN_FILENO, TCSANOW, &newt); // Apply the new settings
	}
	else {
		tcsetattr(STDIN_FILENO, TCSANOW, &oldt); // Restore old settings
	}
}

void setNonBlockingInput(bool enable) {
	int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
	if (enable) {
		fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK); // Modalit� non bloccante
	}
	else {
		fcntl(STDIN_FILENO, F_SETFL, flags & ~O_NONBLOCK); // Ripristina modalit� bloccante
	}
}

void monitorKeypress() {
	setTerminalRawMode(true);
	setNonBlockingInput(true);  // Imposta stdin in modalit� non bloccante

	while (!stopMonitorKey) {
		Timer::SleepMillis(1);
		char ch;
		if (read(STDIN_FILENO, &ch, 1) > 0) { // Ora non blocca
			if (ch == 'p' || ch == 'P') {
				Pause = !Pause;
			}
		}
	}

	setNonBlockingInput(false);  // Ripristina modalit� normale
	setTerminalRawMode(false);
}
#endif


#define RELEASE "2.10 by FixedPaul"

using namespace std;

// ------------------------------------------------------------------------------------------

// Add this function to convert a Bitcoin address to Hash160 hex string
std::string addressToHash160(Secp256K1* secp, const std::string& addr) {
	int addrType = -1;
	
	// Determine address type
	if (addr.length() < 2) {
		return "";
	}
	
	switch (addr.data()[0]) {
	case '1':
		addrType = P2PKH;
		break;
	case '3':
		addrType = P2SH;
		break;
	case 'b':
	case 'B': {
		std::string lowerAddr = addr;
		std::transform(lowerAddr.begin(), lowerAddr.end(), lowerAddr.begin(), ::tolower);
		if (strncmp(lowerAddr.c_str(), "bc1q", 4) == 0)
			addrType = BECH32;
		break;
	}
	default:
		fprintf(stderr, "Error: Unsupported address format (must start with 1, 3, or bc1q)\n");
		return "";
	}
	
	// Convert to Hash160
	std::vector<unsigned char> result;
	if (addrType == BECH32) {
		// BECH32
		uint8_t witprog[40];
		size_t witprog_len;
		int witver;
		const char* hrp = "bc";
		
		if (!segwit_addr_decode(&witver, witprog, &witprog_len, hrp, addr.c_str()) || witprog_len != 20) {
			fprintf(stderr, "Error: Invalid BECH32 address\n");
			return "";
		}
		
		// Convert to hex
		char tmp[41];
		for (int i = 0; i < 20; i++) {
			sprintf(tmp + (i * 2), "%02X", witprog[i]);
		}
		tmp[40] = 0;
		return std::string(tmp);
	} else {
		// P2PKH/P2SH
		if (!DecodeBase58(addr, result) || result.size() != 25) {
			fprintf(stderr, "Error: Invalid Base58 address\n");
			return "";
		}
		
		// Extract hash160 (skip version byte)
		char tmp[41];
		for (int i = 0; i < 20; i++) {
			sprintf(tmp + (i * 2), "%02X", result[i+1]);
		}
		tmp[40] = 0;
		return std::string(tmp);
	}
}

void printUsage() {

	printf("VanitySeacrh [-v] [-gpuId] [-i inputfile] [-o outputfile] [-start HEX] [-end HEX] [-range] [-m] [-stop] [-random] [-hash160] [-convertAddr]\n \n");
	printf(" -v: Print version\n");
	printf(" -i inputfile: Get list of addresses to search from specified file\n");
	printf(" -o outputfile: Output results to the specified file\n");
	printf(" -gpuId: GPU to use, comma-separated list for multiple GPUs, default is 0\n");
	printf(" -start start Private Key HEX\n");
	printf(" -end end Private Key HEX\n");
	printf(" -range bit range dimension. start -> (start + 2^range).\n");
	printf(" -m: Max number of prefixes found by each kernel call, default is 262144 (use multiple of 65536)\n");
	printf(" -stop: Stop when all prefixes are found\n");
	printf(" -random: Random mode active. Each GPU thread scan 1024 random sequentally keys at each step. Not active by default\n");
	printf(" -hash160: Search for Hash160 in hex format instead of Bitcoin addresses\n");
	printf(" -convertAddr: Convert Bitcoin addresses to Hash160 format for more efficient searching\n");
	exit(-1);
}

int getInt(string name, char* v) {

	int r;

	try {

		r = std::stoi(string(v));

	}
	catch (std::invalid_argument&) {

		fprintf(stderr, "[ERROR] Invalid %s argument, number expected\n", name.c_str());
		exit(-1);
	}

	return r;
}

void getInts(string name, vector<int>& tokens, const string& text, char sep) {

	size_t start = 0, end = 0;
	tokens.clear();
	int item;

	try {

		while ((end = text.find(sep, start)) != string::npos) {
			item = std::stoi(text.substr(start, end - start));
			tokens.push_back(item);
			start = end + 1;
		}

		item = std::stoi(text.substr(start));
		tokens.push_back(item);

	}
	catch (std::invalid_argument&) {

		fprintf(stderr, "[ERROR] Invalid %s argument, number expected\n", name.c_str());
		exit(-1);
	}
}

void getKeySpace(const string& text, BITCRACK_PARAM* bc, Int& maxKey)
{
	size_t start = 0, end = 0;
	string item;

	try
	{
		if ((end = text.find(':', start)) != string::npos)
		{
			item = std::string(text.substr(start, end));
			start = end + 1;
		}
		else
		{
			item = std::string(text);
		}

		if (item.length() == 0)
		{
			bc->ksStart.SetInt32(1);
		}
		else if (item.length() > 64)
		{
			fprintf(stderr, "[ERROR] keyspaceSTART: invalid privkey (64 length)\n");
			exit(-1);
		}
		else
		{
			item.insert(0, 64 - item.length(), '0');
			for (int i = 0; i < 32; i++)
			{
				unsigned char my1ch = 0;
				if (sscanf(&item[2 * i], "%02hhX", &my1ch)) {};
				bc->ksStart.SetByte(31 - i, my1ch);
			}
		}

		if (start != 0 && (end = text.find('+', start)) != string::npos)
		{
			item = std::string(text.substr(end + 1));
			if (item.length() > 64 || item.length() == 0)
			{
				fprintf(stderr, "[ERROR] keyspace__END: invalid privkey (64 length)\n");
				exit(-1);
			}

			item.insert(0, 64 - item.length(), '0');

			for (int i = 0; i < 32; i++)
			{
				unsigned char my1ch = 0;
				if (sscanf(&item[2 * i], "%02hhX", &my1ch)) {};
				bc->ksFinish.SetByte(31 - i, my1ch);
			}

			bc->ksFinish.Add(&bc->ksStart);
		}
		else if (start != 0)
		{
			item = std::string(text.substr(start));

			if (item.length() > 64 || item.length() == 0)
			{
				fprintf(stderr, "[ERROR] keyspace__END: invalid privkey (64 length)\n");
				exit(-1);
			}

			item.insert(0, 64 - item.length(), '0');

			for (int i = 0; i < 32; i++)
			{
				unsigned char my1ch = 0;
				if (scanf(&item[2 * i], "%02hhX", &my1ch)) {};
				bc->ksFinish.SetByte(31 - i, my1ch);
			}
		}
		else
		{
			bc->ksFinish.Set(&maxKey);
		}
	}
	catch (std::invalid_argument&)
	{
		fprintf(stderr, "[ERROR] Invalid --keyspace argument \n");
		exit(-1);
	}
}

void checkKeySpace(BITCRACK_PARAM* bc, Int& maxKey)
{
	if (bc->ksStart.IsGreater(&maxKey) || bc->ksFinish.IsGreater(&maxKey))
	{
		fprintf(stderr, "[ERROR] START/END IsGreater %s \n", maxKey.GetBase16().c_str());
		exit(-1);
	}

	if (bc->ksFinish.IsLowerOrEqual(&bc->ksStart))
	{
		fprintf(stderr, "[ERROR] END IsLowerOrEqual START \n");
		exit(-1);
	}

	if (bc->ksFinish.IsLowerOrEqual(&bc->ksNext))
	{
		fprintf(stderr, "[ERROR] END: IsLowerOrEqual NEXT \n");
		exit(-1);
	}

	return;
}

void parseFile(string fileName, vector<string>& lines) {

	// Get file size
	FILE* fp = fopen(fileName.c_str(), "rb");
	if (fp == NULL) {
		fprintf(stderr, "[ERROR] ParseFile: cannot open %s %s\n", fileName.c_str(), strerror(errno));
		exit(-1);
	}
	fseek(fp, 0L, SEEK_END);
	size_t sz = ftell(fp);
	size_t nbAddr = sz / 33; /* Upper approximation */
	bool loaddingProgress = sz > 100000;
	fclose(fp);

	// Parse file
	int nbLine = 0;
	string line;
	ifstream inFile(fileName);
	lines.reserve(nbAddr);
	while (getline(inFile, line)) {

		// Remove ending \r\n
		int l = (int)line.length() - 1;
		while (l >= 0 && isspace(line.at(l))) {
			line.pop_back();
			l--;
		}

		if (line.length() > 0) {
			lines.push_back(line);
			nbLine++;
			if (loaddingProgress) {
				if ((nbLine % 50000) == 0)
					fprintf(stdout, "[Loading input file %5.1f%%]\r", ((double)nbLine * 100.0) / ((double)(nbAddr) * 33.0 / 34.0));
			}
		}
	}

	if (loaddingProgress)
		fprintf(stdout, "[Loading input file 100.0%%]\n");
}

void generateKeyPair(Secp256K1* secp, string seed, int searchMode, bool paranoiacSeed) {

	if (seed.length() < 8) {
		fprintf(stderr, "[ERROR] Use a seed of at leats 8 characters to generate a key pair\n");
		fprintf(stderr, "Ex: VanitySearch -s \"A Strong Password\" -kp\n");
		exit(-1);
	}

	if (searchMode == SEARCH_BOTH) {
		fprintf(stderr, "[ERROR] Use compressed or uncompressed to generate a key pair\n");
		exit(-1);
	}

	bool compressed = (searchMode == SEARCH_COMPRESSED);

	string salt = "0";
	unsigned char hseed[64];
	pbkdf2_hmac_sha512(hseed, 64, (const uint8_t*)seed.c_str(), seed.length(),
		(const uint8_t*)salt.c_str(), salt.length(),
		2048);

	Int privKey;
	privKey.SetInt32(0);
	sha256(hseed, 64, (unsigned char*)privKey.bits64);
	Point p = secp->ComputePublicKey(&privKey);
	fprintf(stdout, "Priv : %s\n", secp->GetPrivAddress(compressed, privKey).c_str());
	fprintf(stdout, "Pub  : %s\n", secp->GetPublicKeyHex(compressed, p).c_str());
}

void outputAdd(string outputFile, int addrType, string addr, string pAddr, string pAddrHex) {

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

	fprintf(f, "\nPublic Addr: %s\n", addr.c_str());

	switch (addrType) {
	case P2PKH:
		fprintf(f, "Priv (WIF): p2pkh:%s\n", pAddr.c_str());
		break;
	case P2SH:
		fprintf(f, "Priv (WIF): p2wpkh-p2sh:%s\n", pAddr.c_str());
		break;
	case BECH32:
		fprintf(f, "Priv (WIF): p2wpkh:%s\n", pAddr.c_str());
		break;
	}
	fprintf(f, "Priv (HEX): 0x%s\n", pAddrHex.c_str());

	if (needToClose)
		fclose(f);
}

#define CHECK_ADDR()                                           \
  fullPriv.ModAddK1order(&e, &partialPrivKey);                 \
  p = secp->ComputePublicKey(&fullPriv);                       \
  cAddr = secp->GetAddress(addrType, compressed, p);           \
  if (cAddr == addr) {                                         \
    found = true;                                              \
    string pAddr = secp->GetPrivAddress(compressed, fullPriv); \
    string pAddrHex = fullPriv.GetBase16();                    \
    outputAdd(outputFile, addrType, addr, pAddr, pAddrHex);    \
  }

void reconstructAdd(Secp256K1* secp, string fileName, string outputFile, string privAddr) {

	bool compressed;
	int addrType;
	Int lambda;
	Int lambda2;
	lambda.SetBase16("5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72");
	lambda2.SetBase16("ac9c52b33fa3cf1f5ad9e3fd77ed9ba4a880b9fc8ec739c2e0cfc810b51283ce");

	Int privKey = secp->DecodePrivateKey((char*)privAddr.c_str(), &compressed);
	if (privKey.IsNegative())
		exit(-1);

	vector<string> lines;
	parseFile(fileName, lines);

	for (int i = 0; i < (int)lines.size(); i += 2) {

		string addr;
		string partialPrivAddr;

		if (lines[i].substr(0, 10) == "Pub Addr: ") {

			addr = lines[i].substr(10);

			switch (addr.data()[0]) {
			case '1':
				addrType = P2PKH; break;
			case '3':
				addrType = P2SH; break;
			case 'b':
			case 'B':
				addrType = BECH32; break;
			default:
				printf("Invalid partialkey info file at line %d\n", i);
				printf("%s Address format not supported\n", addr.c_str());
				continue;
			}

		}
		else {
			printf("[ERROR] Invalid partialkey info file at line %d (\"Pub Addr: \" expected)\n", i);
			exit(-1);
		}

		if (lines[i + 1].substr(0, 13) == "PartialPriv: ") {
			partialPrivAddr = lines[i + 1].substr(13);
		}
		else {
			printf("[ERROR] Invalid partialkey info file at line %d (\"PartialPriv: \" expected)\n", i);
			exit(-1);
		}

		bool partialMode;
		Int partialPrivKey = secp->DecodePrivateKey((char*)partialPrivAddr.c_str(), &partialMode);
		if (privKey.IsNegative()) {
			printf("[ERROR] Invalid partialkey info file at line %d\n", i);
			exit(-1);
		}

		if (partialMode != compressed) {

			printf("[WARNING] Invalid partialkey at line %d (Wrong compression mode, ignoring key)\n", i);
			continue;

		}
		else {

			// Reconstruct the address
			Int fullPriv;
			Point p;
			Int e;
			string cAddr;
			bool found = false;

			// No sym, no endo
			e.Set(&privKey);
			CHECK_ADDR();


			// No sym, endo 1
			e.Set(&privKey);
			e.ModMulK1order(&lambda);
			CHECK_ADDR();

			// No sym, endo 2
			e.Set(&privKey);
			e.ModMulK1order(&lambda2);
			CHECK_ADDR();

			// sym, no endo
			e.Set(&privKey);
			e.Neg();
			e.Add(&secp->order);
			CHECK_ADDR();

			// sym, endo 1
			e.Set(&privKey);
			e.ModMulK1order(&lambda);
			e.Neg();
			e.Add(&secp->order);
			CHECK_ADDR();

			// sym, endo 2
			e.Set(&privKey);
			e.ModMulK1order(&lambda2);
			e.Neg();
			e.Add(&secp->order);
			CHECK_ADDR();

			if (!found) {
				printf("Unable to reconstruct final key from partialkey line %d\n Addr: %s\n PartKey: %s\n",
					i, addr.c_str(), partialPrivAddr.c_str());
			}
		}
	}
}

int main(int argc, char* argv[]) {

	std::thread inputThread(monitorKeypress);

	// Global Init
	Timer::Init();

	// Init SecpK1
	Secp256K1* secp = new Secp256K1();
	secp->Init();

	// Browse arguments
	if (argc < 2) {
		fprintf(stderr, "Not enough argument\n");
		printUsage();
		exit(-1);
	}

	int a = 1;
	bool stop = false;
	int searchMode = SEARCH_COMPRESSED;
	bool useHash160 = false;  // Flag to indicate we're in Hash160 search mode
	bool convertAddresses = false;  // Flag to indicate if addresses should be converted to hash160
	vector<int> gpuId = { 0 };
	string gpuParsed = "0";
	vector<int> gridSize;
	vector<string> address;
	string outputFile = "";
	uint32_t maxFound = 65536*4;
	int range = 30;
	string start = "0";
	
	// bitcrack mod
	BITCRACK_PARAM bitcrack, *bc;
	bc = &bitcrack;
	Int maxKey;
		
	maxKey.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140");
		
	bc->ksStart.SetInt32(1);
	bc->ksNext.Set(&bc->ksStart);
	bc->ksFinish.Set(&maxKey);
	bc->useEndKey = false;

	while (a < argc) {

		if (strcmp(argv[a], "-gpuId") == 0) {
			a++;
			gpuParsed = string(argv[a]);
			a++;
		}
		else if (strcmp(argv[a], "-v") == 0) {
			printf("%s\n", RELEASE);
			exit(0);

		}
		else if (strcmp(argv[a], "-o") == 0) {
			a++;
			outputFile = string(argv[a]);
			a++;
		}
		else if (strcmp(argv[a], "-start") == 0) {
			a++;
			start = string(argv[a]);
			a++;
		}
		else if (strcmp(argv[a], "-end") == 0) {
			a++;
			getKeySpace(string(start + ":" + string(argv[a])), bc, maxKey);
			bc->useEndKey = true;
			a++;
		}
		else if (strcmp(argv[a], "-i") == 0) {
			a++;
			parseFile(string(argv[a]), address);
			a++;

		}
		else if (strcmp(argv[a], "-stop") == 0) {
			stop = true;
			a++;
		}
		else if (strcmp(argv[a], "-random") == 0) {
			randomMode = true;
			a++;
		}
		else if (strcmp(argv[a], "-hash160") == 0) {
			useHash160 = true;  // Enable Hash160 search mode
			a++;
		}
		else if (strcmp(argv[a], "-convertAddr") == 0) {
			convertAddresses = true;  // Convert addresses to hash160
			a++;
		}
		else if (strcmp(argv[a], "-range") == 0) {
			a++;
			range = (uint64_t)getInt("range", argv[a]);
			a++;
		}
		else if (strcmp(argv[a], "-m") == 0) {
			a++;
			maxFound = getInt("maxFound", argv[a]);
			a++;
		}
		else if (a == argc - 1) {
			address.push_back(string(argv[a]));
			a++;
		}
		else {
			printf("Unexpected %s argument\n", argv[a]);
			printUsage();
		}

	}

	fprintf(stdout, "VanitySearch-Bitcrack v" RELEASE "\n");

	// Parse comma-separated GPU IDs
	std::vector<int> gpuIds = parseGPUIds(gpuParsed);
	int numGPUs = (int)gpuIds.size();

	if (gridSize.size() == 0) {
		for (int i = 0; i < numGPUs; i++) {
			gridSize.push_back(-1);
			gridSize.push_back(128);
		}
	}
	else if (gridSize.size() != gpuIds.size() * 2) {
		printf("Invalid gridSize or gpuId argument, must have coherent size\n");
		exit(-1);
	}

	// Convert any provided Bitcoin addresses to Hash160 format if requested
	if (convertAddresses) {
		useHash160 = true;  // Force Hash160 search mode
		vector<string> hash160Addresses;
		
		for (int i = 0; i < address.size(); i++) {
			string hash160 = addressToHash160(secp, address[i]);
			if (!hash160.empty()) {
				printf("Converting %s to Hash160: %s\n", address[i].c_str(), hash160.c_str());
				hash160Addresses.push_back(hash160);
			} else {
				fprintf(stderr, "Failed to convert address: %s\n", address[i].c_str());
			}
		}
		
		// Replace the address list with the Hash160 versions
		if (!hash160Addresses.empty()) {
			address = hash160Addresses;
		} else if (!address.empty()) {
			fprintf(stderr, "No valid addresses to convert to Hash160. Exiting.\n");
			exit(-1);
		}
	}

	// Set search type to HASH160 if we're using Hash160 search mode
	if (useHash160) {
		searchMode = SEARCH_COMPRESSED;  // We'll still compress keys for efficiency
		for (int i = 0; i < address.size(); i++) {
			// Ensure address is properly formatted for Hash160 search
			if (address[i].length() < 40) {
				// Pad with zeros to make it valid length if needed
				address[i].insert(0, 40 - address[i].length(), '0');
			}
			else if (address[i].length() > 40) {
				printf("Warning: Hash160 value '%s' is too long, using first 40 characters\n", address[i].c_str());
				address[i] = address[i].substr(0, 40);
			}
		}
	}

	if (range > 255)
		range = 255;

	// Calculate range if -end parameter wasn't used
	if (strcmp(argv[a-1], "-end") != 0) {
		Int Range;
		Range.SetInt32(1);

		for (int i = 0; i < range; i++) {
			Range.Mult(2);
		}
		Range.SubOne();

		getKeySpace(string(start + ":+" + Range.GetBase16()), bc, maxKey);
		bc->ksNext.Set(&bc->ksStart);
		checkKeySpace(bc, maxKey);
	}

	{
		fprintf(stdout, "[keyspace]  start=%s\n", bc->ksStart.GetBase16().c_str());
		fprintf(stdout, "[keyspace]    end=%s\n", bc->ksFinish.GetBase16().c_str());
		if (randomMode) {
			fprintf(stdout, "Random Mode\n");
		}
		if (useHash160) {
			fprintf(stdout, "Hash160 Direct Search Mode\n");
		}
		fprintf(stdout, "Using %d GPU(s)\n", numGPUs);
		fflush(stdout);


		idxcount = 0;
		t_Paused = 0;
		Pause = false;
	repeatP:
		Paused = false;
		VanitySearch* v = new VanitySearch(secp, address, searchMode, stop, outputFile, maxFound, bc);

		// Set search type to HASH160 if using that mode
		if (useHash160) {
			v->SetSearchType(HASH160);
		}

		v->Search(gpuIds, gridSize);

		while (Paused) {
			Timer::SleepMillis(100);
			if (!Pause) {
				fprintf(stdout, "\nResuming...\n");
				goto repeatP;
				
			}
		}
	}

	stopMonitorKey = true;

	Timer::SleepMillis(100);

	if (inputThread.joinable()) {
		inputThread.join();
	}

	return 0;
}

// Parse comma-separated GPU IDs
std::vector<int> parseGPUIds(const std::string& gpuParsed) {
	std::vector<int> gpuIds;
	std::string s = gpuParsed;
	std::string delimiter = ",";
	size_t pos = 0;
	std::string token;
	
	while ((pos = s.find(delimiter)) != std::string::npos) {
		token = s.substr(0, pos);
		gpuIds.push_back(std::stoi(token));
		s.erase(0, pos + delimiter.length());
	}
	
	// Add the last GPU id
	if (!s.empty()) {
		gpuIds.push_back(std::stoi(s));
	}
	
	return gpuIds;
}
