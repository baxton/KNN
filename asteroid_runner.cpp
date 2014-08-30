
//
// g++ runner.cpp -o runner.exe
//

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>

using namespace std;


int main(int argc, const char* argv[]) {
    string line;

    // feed train
    for (int i = 1; i < 6; ++i) {
        stringstream ss;
        ss << "C:\\Temp\\asteroid\\data2\\train_" << i << ".txt";

        ifstream ifs;
        ifs.open(ss.str().c_str(), ios_base::in);

        for (; std::getline(ifs, line); ) {
            cout << line << endl;
        }

        //std::getline(cin, line);
        //int result = atoi(line.c_str());
        //if (result == 1)
        //    break;
    }
/*
    // feed test
    for (int i = 1; i < 21; ++i) {
        stringstream ss;
        ss << "C:\\Temp\\asteroid\\data2\\test_" << i << ".txt";

        ifstream ifs;
        ifs.open(ss.str().c_str(), ios_base::in);

        for (; std::getline(ifs, line); ) {
            cout << line << endl;
        }

        std::getline(cin, line);
        int result = atoi(line.c_str());
    }

    // read result
    vector<string> result;
    std::getline(cin, line);
    int size = atoi(line.c_str());

    for (int i = 0; i < size; ++i) {
        std::getline(cin, line);
        result.push_back(line);
    }
*/
    return 0;
}
