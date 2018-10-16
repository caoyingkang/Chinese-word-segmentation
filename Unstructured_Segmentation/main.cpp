#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include "Unstr_CWS.h"

using namespace std;

int main()
{
    Unstr_CWS cws("../data/train.txt", "unstruct.log");
    cws.showFeatures("feat.unstruct.txt");
    cws("..data/test.txt", "result.unstruct.txt");
    return 0;
}
