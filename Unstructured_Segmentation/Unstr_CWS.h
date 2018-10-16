#ifndef UNSTR_CWS_H_INCLUDED
#define UNSTR_CWS_H_INCLUDED

#include <set>
#include <vector>
#include <map>

//the type of feature based on numbers
using fType = std::size_t;

//the type of the Feature Map
//key : the feature based on numbers
//value : the index of the corresponding element in weight vector
using FmapType = std::map<fType, std::size_t>;

//the type of feature vector
//the element of feature vector is size_t which denotes the index of this feature in weight vector
using fVec = std::vector<std::size_t>;

//The Byte Order Mark appearing at the beginning of .txt file
const std::string BOM = {char(0xEF), char(0xBB), char(0xBF)};

//a set of punctuations:
const std::set<std::string> Puncs = {
    {char(0xE2), char(0x80), char(0x9C)},
    {char(0xE2), char(0x80), char(0x9D)},
    {char(0xE2), char(0x80), char(0x98)},
    {char(0xE2), char(0x80), char(0x99)},
    {char(0xE3), char(0x80), char(0x81)},
    {char(0xEF), char(0xBC), char(0x8C)},
    {char(0xE3), char(0x80), char(0x82)},
    {char(0xEF), char(0xBC), char(0x9B)},
    {char(0xEF), char(0xBC), char(0x9A)},
    {char(0xEF), char(0xBC), char(0x88)},
    {char(0xEF), char(0xBC), char(0x89)},
    {char(0xE3), char(0x80), char(0x8A)},
    {char(0xE3), char(0x80), char(0x8B)},
    {char(0xEF), char(0xBC), char(0x81)},
    {char(0xEF), char(0xBC), char(0x9F)},
    {char(0xE2), char(0x80), char(0x94)},
    {char(0xE2), char(0x80), char(0xA6)}};

inline bool isPunc(const std::string &ch)
{
    return Puncs.find(ch) != Puncs.end();
}

inline bool isSpace(const std::string &ch)
{
    return ch == " ";
}

/*
Change bits from utf8-format to a set of characters, where each character is stored as a string;
return the size of the set(how many characters are taken out)
*/
void utf8_to_chrcs(const std::string &input, std::vector<std::string> &output);

class Unstr_CWS
{
  public:
    Unstr_CWS(const std::string &_train_file, const std::string &_log_file);
    ~Unstr_CWS() { log.close(); }
    void showFeatures(const std::string &_feature_file) const;
    void operator()(const std::string &_test_file, const std::string &_result_file) const;

  private:
    void col_F_from_stdfile();
    void stdsent_to_puresent_with_stdtags(std::vector<std::string>::const_iterator b, std::vector<std::string>::const_iterator e,
                                          std::vector<std::string> &puresent, std::string &stdtags) const;
    void col_F_from_puresent_with_stdtags(const std::vector<std::string> &puresent, const std::string &stdtags, std::ofstream &os);
    inline std::size_t add_Unigram_feature(const std::string &ch, char tag, char pos);
    inline std::size_t add_Bigram_feature(const std::string &ch1, const std::string &ch2, char tag, char pos);
    void opt_w_from_stdfile();
    fVec segment_f(const std::vector<std::string> &puresent) const;
    void avr_segment(const std::vector<std::string> &puresent, std::string &mytags) const;
    void fVec_add(fVec &f, const std::string &str) const;
    std::pair<int, fVec> chrc_score_Int_f(const std::string &ch, const std::string &pre_ch, const std::string &post_ch, char tag) const;
    double chrc_score_Double(const std::string &ch, const std::string &pre_ch, const std::string &post_ch, char tag) const;
    int score_Int(const fVec &f) const;
    double score_Double(const fVec &f) const;
    bool cmp_f(fVec &stdf, fVec &myf, fVec &addf, fVec &minusf) const;

    std::string train_file;
    std::string pure_train_file = "pure_train_file.txt";
    FmapType Fmap;
    std::vector<int> w;        //weight vector
    std::vector<int> sum_w;    //sum weight vector
    std::vector<double> avr_w; //averaged weight vector
    std::ofstream log;         //used for outputting information when debugging
};

#endif // UNSTR_CWS_H_INCLUDED
