#ifndef CWS_H_INCLUDED
#define CWS_H_INCLUDED

#include <set>
#include <vector>
#include <map>

/* type of feature (we use number of counts as feature) */
using fType = std::size_t;

/* type of feature map.
   key : feature
   value : index denoting which element in weight vector corresponds this feature */
using FmapType = std::map<fType, std::size_t>;

/* the type of feature vector (a vector of features).
   each element denotes the index of this feature in weight vector */
using fVec = std::vector<std::size_t>;

/* Byte Order Mark appearing at the beginning of .txt file: 0xEFBBBF */
const std::string BOM = {char(0xEF), char(0xBB), char(0xBF)};

/* set of punctuations */
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

/* convert bytes from utf8 to characters: each character is stored as a string; */
void utf8_to_chrcs(const std::string &input, std::vector<std::string> &output);

class CWS
{
  private:
    std::string train_file;
    std::string pure_train_file = "pure_train_file.txt";
    FmapType Fmap;
    std::vector<int> w;        //weight vector
    std::vector<int> sum_w;    //accumulated weight vector
    std::vector<double> avr_w; //averaged weight vector
    std::ofstream log;         //used for outputing infor when debugging

  public:
    CWS(const std::string &_train_file, const std::string &_log_file);
    ~CWS() { log.close(); }
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
};

struct Vertex
{
    Vertex(std::size_t i)
    {
        if (i == 0)
        {
            cut = true;
            pre_c = pre_n = pre = -1; //a very large number, a symbol for beginning
        }
        else if (i == 1)
        {
            cut = false;
            pre_c = pre_n = pre = -1;
        }
        else if (i % 2)
        {
            cut = false;
            pre_c = i - 3;
            pre_n = i - 2;
        }
        else
        {
            cut = true;
            pre_c = i - 2;
            pre_n = i - 1;
        }
    }
    bool cut;
    std::size_t pre_c;
    std::size_t pre_n;
    std::size_t pre;
};

struct Vertex_f : public Vertex
{
    Vertex_f(std::size_t i) : Vertex(i) {}

    int scr;
    fVec f;
};

struct Vertex_avr : public Vertex
{
    Vertex_avr(std::size_t i) : Vertex(i) {}

    double scr;
};

#endif // CWS_H_INCLUDED
