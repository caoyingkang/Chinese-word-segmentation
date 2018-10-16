#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include "CWS.h"

using namespace std;

int main()
{
    CWS cws("../data/train.txt", "struct.log");
    cws.showFeatures("feat.struct.txt");
    cws("../data/test.txt", "result.struct.txt");
    return 0;
}
