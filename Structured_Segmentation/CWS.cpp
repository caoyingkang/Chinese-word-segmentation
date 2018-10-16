#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <functional>
#include <cassert>
#include <utility>
#include <algorithm>
#include <ctime>
#include "CWS.h"

// #define FEATURE_MAP

using namespace std;

hash<string> str_hash;

/* convert bytes from utf8 to characters: each character is stored as a string; */
void utf8_to_chrcs(const string &input, vector<string> &output)
{
    output.clear(); //clear before store
    string chrc;
    for (size_t i = 0, len = 0; i < input.size(); i += len)
    {
        unsigned char first_byte = unsigned(input[i]); //the first byte of each character
        if (first_byte >= 0xFC)
            len = 6;
        else if (first_byte >= 0xF8)
            len = 5;
        else if (first_byte >= 0xF0)
            len = 4;
        else if (first_byte >= 0xE0)
            len = 3;
        else if (first_byte >= 0xC0)
            len = 2;
        else
            len = 1;
        chrc = input.substr(i, len);
        if (chrc == BOM)
            continue;
        output.push_back(chrc);
    }
}

#ifdef FEATURE_MAP //obtain the feature map directly from "feature_map.txt"
CWS::CWS(const std::string &_train_file, const std::string &_log_file)
    : train_file(_train_file), log(_log_file)
{

    //obtain the feature map
    ifstream is("feature_map.txt");
    size_t f, s;
    while (is >> f >> s)
    {
        Fmap.insert(make_pair(f, s));
    }
    is.close();

    w.assign(Fmap.size(), 0);
    sum_w.assign(Fmap.size(), 0);
    avr_w.assign(Fmap.size(), 0.0);

    cout << "Collect features completed!" << endl;

    //optimize weight vector
    opt_w_from_stdfile();
}
#else  //extract feature map from train file
CWS::CWS(const std::string &_train_file, const std::string &_log_file)
    : train_file(_train_file), log(_log_file)
{
    /* extract the feature map and 
       output the pure train file (used for further learning) */
    col_F_from_stdfile();
    cout << "Collect features completed!" << endl;

    // optimize weight vector
    opt_w_from_stdfile();
}
#endif // FEATURE_SET

void CWS::showFeatures(const string &_feature_file) const
{
    ofstream os(_feature_file);
    for (auto f : Fmap)
        os << f.first << " " << f.second << endl;
    os.close();
}

//collect all the features in training file;
//and store them in Fmap
void CWS::col_F_from_stdfile()
{
    ifstream is(train_file);
    ofstream os(pure_train_file);
    string raw_line;         //a line of sentences scanned from a utf8 txt
    vector<string> std_line; //a line of characters
    vector<string> puresent; //a sentence with no spaces or punctuations
    string stdtags;

    while (getline(is, raw_line))
    {
        utf8_to_chrcs(raw_line, std_line);
        for (auto it1 = std_line.begin(), it2 = std_line.begin(); it1 != std_line.end();)
        {
            while (isSpace(*it1) || isPunc(*it1))
            {
                ++it1;
                if (it1 == std_line.end())
                    break;
            }
            if (it1 == std_line.end())
                break;
            it2 = it1 + 1;
            while (it2 != std_line.end() && !isPunc(*it2))
            {
                ++it2;
            }

            stdsent_to_puresent_with_stdtags(it1, it2, puresent, stdtags);
            col_F_from_puresent_with_stdtags(puresent, stdtags, os);

            if (it2 == std_line.end())
                break;
            else
                it1 = it2 + 1;
        }
    }
    is.close();
    os.close();
}

//convert a standard sentence(denoted by range b and e)
//into a pure sentence with its tags
//(b is promised to denote a character)
void CWS::stdsent_to_puresent_with_stdtags(vector<string>::const_iterator b, vector<string>::const_iterator e,
                                           vector<string> &puresent, string &stdtags) const
{
    puresent.clear();
    stdtags.clear();
    bool flag = false; //when flag is true, last character denoted by "it" is not a space
    assert(!isSpace(*b));
    auto it = b;
    while (it != e)
    {
        if (!isSpace(*it))
        {
            puresent.push_back(*it);
            if (flag)
                stdtags.push_back('0');
            else
                stdtags.push_back('1');
            flag = true;
        }
        else if (flag)
        {
            stdtags[stdtags.size() - 1] += 2;
            flag = false;
        }
        ++it;
    }
    if (flag)
    {
        stdtags[stdtags.size() - 1] += 2;
    }
}

//collect features from a pure sentence and its standard tags
//and output the pure sentence with its sorted feature vector into file "pure_train_file.txt"
void CWS::col_F_from_puresent_with_stdtags(const std::vector<std::string> &puresent, const std::string &stdtags, ofstream &os)
{
    assert(puresent.size() == stdtags.size());
    for (auto &r : puresent)
    {
        os << r;
    }
    os << endl;
    size_t sz = puresent.size();
    fVec f;
    for (size_t i = 0; i < sz; ++i)
    {
        f.push_back(add_Unigram_feature(puresent[i], stdtags[i], 'C'));
        if (i > 0)
        {
            f.push_back(add_Unigram_feature(puresent[i - 1], stdtags[i], 'L'));
            f.push_back(add_Bigram_feature(puresent[i - 1], puresent[i], stdtags[i], 'L'));
        }
        if (i < sz - 1)
        {
            f.push_back(add_Unigram_feature(puresent[i + 1], stdtags[i], 'R'));
            f.push_back(add_Bigram_feature(puresent[i], puresent[i + 1], stdtags[i], 'R'));
        }
        if (i > 0 && i < sz - 1)
        {
            f.push_back(add_Bigram_feature(puresent[i - 1], puresent[i + 1], stdtags[i], 'C'));
        }
    }
    sort(f.begin(), f.end());
    for (auto &r : f)
    {
        os << r << ' ';
    }
    os << endl;
}

//add Unigram feature into Fmap
//and return the index of this feature in w(no mater this feature is whether new to the Fmap)
inline size_t CWS::add_Unigram_feature(const std::string &ch, char tag, char pos)
{
    fType feat = str_hash(ch + '_' + tag + pos);
    auto re = Fmap.insert(make_pair(feat, w.size()));
    if (re.second)
    { //successfully inserted
        w.push_back(0);
        sum_w.push_back(0);
        avr_w.push_back(0.0);
    }
    return re.first->second;
}

//add Bigram feature into Fmap
//and return the index of this feature in w(no mater this feature is whether new to the Fmap)
inline size_t CWS::add_Bigram_feature(const std::string &ch1, const std::string &ch2, char tag, char pos)
{
    fType feat = str_hash(ch1 + '_' + ch2 + '_' + tag + pos);
    auto re = Fmap.insert(make_pair(feat, w.size()));
    if (re.second)
    { //successfully inserted
        w.push_back(0);
        sum_w.push_back(0);
        avr_w.push_back(0.0);
    }
    return re.first->second;
}

void CWS::opt_w_from_stdfile()
{
    string raw_line;
    vector<string> puresent; //a sentence containing no spaces or punctuations
    fVec stdf, myf, addf, minusf;

    size_t max_n = 0;
    vector<size_t> last_t(w.size(), 0);
    vector<size_t> last_n(w.size(), 0);

    for (size_t t = 0; t < 12; ++t)
    { //one traverse of the pure_train_file for each iteration in "for"
        size_t n = 0;
        size_t wrong_count = 0;
        ifstream is(pure_train_file);
        while (getline(is, raw_line))
        { //one sentence for each iteration in "while"
            //obtain puresent and stdf
            utf8_to_chrcs(raw_line, puresent);
            getline(is, raw_line);
            istringstream ss(raw_line);
            stdf.clear();
            fType feat;
            while (ss >> feat)
            {
                stdf.push_back(feat);
            }

            //get myf, addf, and minusf
            myf = std::move(segment_f(puresent));

            //optimize weight vector

            if (!cmp_f(stdf, myf, addf, minusf))
            {
                wrong_count += addf.size();
                for (auto &index : addf)
                {
                    sum_w[index] += w[index] * ((t - last_t[index]) * max_n + n - last_n[index]) + 1;
                    ++w[index];
                    last_t[index] = t;
                    last_n[index] = n;
                }
                for (auto &index : minusf)
                {
                    sum_w[index] += w[index] * ((t - last_t[index]) * max_n + n - last_n[index]) - 1;
                    --w[index];
                    last_t[index] = t;
                    last_n[index] = n;
                }
            }

            ++n;
        }
        max_n = n;
        is.close();

        //get avr_w
        for (size_t index = 0; index < w.size(); ++index)
        {
            sum_w[index] += w[index] * ((t - last_t[index]) * max_n + (n - 1) - last_n[index]);
            last_t[index] = t;
            last_n[index] = n - 1;
        }
        auto it1 = sum_w.begin();
        auto it2 = avr_w.begin();
        while (it1 != sum_w.end())
        {
            *it2 = double(*it1) / double((t + 1) * max_n);
            ++it1;
            ++it2;
        }
    }
}

//segmentation with a pure sentence (at least one character)
//return the feature vector
//use w
fVec CWS::segment_f(const vector<string> &puresent) const
{
    fVec f;
    size_t len = puresent.size();

    if (len == 1)
    { //if there is only one character
        fVec_add(f, puresent[0] + '_' + '3' + 'C');
        return std::move(f);
    }

    //if there are at least two characters
    //construct the graph and make optimization
    size_t n = 2 * len - 1; //n>=3
    vector<Vertex_f> graph;
    {
        graph.emplace_back(0);
        auto re = std::move(chrc_score_Int_f(puresent[0], string(), puresent[1], '3'));
        graph[0].scr = re.first;
        graph[0].f = std::move(re.second);
    }
    {
        graph.emplace_back(1);
        auto re = std::move(chrc_score_Int_f(puresent[0], string(), puresent[1], '1'));
        graph[1].scr = re.first;
        graph[1].f = std::move(re.second);
    }
    for (size_t i = 2; i < n - 1; ++i)
    {
        graph.emplace_back(i);
        size_t index = i / 2;
        char tag = (graph[i].cut ? '2' : '0');
        auto re_c = std::move(chrc_score_Int_f(puresent[index], puresent[index - 1], puresent[index + 1], tag + 1));
        auto re_n = std::move(chrc_score_Int_f(puresent[index], puresent[index - 1], puresent[index + 1], tag));
        auto scr_c = re_c.first + graph[graph[i].pre_c].scr;
        auto scr_n = re_n.first + graph[graph[i].pre_n].scr;
        if (scr_c > scr_n)
        {
            graph[i].pre = graph[i].pre_c;
            graph[i].scr = scr_c;
            graph[i].f = std::move(re_c.second);
        }
        else
        {
            graph[i].pre = graph[i].pre_n;
            graph[i].scr = scr_n;
            graph[i].f = std::move(re_n.second);
        }
    }
    {
        graph.emplace_back(n - 1);
        auto re_c = std::move(chrc_score_Int_f(puresent[len - 1], puresent[len - 2], string(), '3'));
        auto re_n = std::move(chrc_score_Int_f(puresent[len - 1], puresent[len - 2], string(), '2'));
        auto scr_c = re_c.first + graph[graph[n - 1].pre_c].scr;
        auto scr_n = re_n.first + graph[graph[n - 1].pre_n].scr;
        if (scr_c > scr_n)
        {
            graph[n - 1].pre = graph[n - 1].pre_c;
            graph[n - 1].scr = scr_c;
            graph[n - 1].f = std::move(re_c.second);
        }
        else
        {
            graph[n - 1].pre = graph[n - 1].pre_n;
            graph[n - 1].scr = scr_n;
            graph[n - 1].f = std::move(re_n.second);
        }
    }

    //back tracking
    size_t i = n - 1;
    while (i != -1)
    {
        f.insert(f.end(), graph[i].f.begin(), graph[i].f.end());
        i = graph[i].pre;
    }
    return std::move(f);
}

//if the feat hashed from str is in the gross feature map,
//add the feat into fVec
void CWS::fVec_add(fVec &f, const string &str) const
{
    fType feat = str_hash(str);
    auto it = Fmap.find(feat);
    if (it != Fmap.cend())
    { //the feature is in the gross feature map
        f.push_back(it->second);
    }
}

//returning the score and its feature vector of one character according to
//its left and right character(if none, assign them with empty string)
//and its tag
//use w
pair<int, fVec> CWS::chrc_score_Int_f(const string &ch, const string &pre_ch, const string &post_ch, char tag) const
{
    fVec f;
    fVec_add(f, ch + '_' + tag + 'C');
    if (pre_ch.size())
    {
        fVec_add(f, pre_ch + '_' + tag + 'L');
        fVec_add(f, pre_ch + '_' + ch + '_' + tag + 'L');
    }
    if (post_ch.size())
    {
        fVec_add(f, post_ch + '_' + tag + 'R');
        fVec_add(f, ch + '_' + post_ch + '_' + tag + 'R');
    }
    if (pre_ch.size() && post_ch.size())
    {
        fVec_add(f, pre_ch + '_' + post_ch + '_' + tag + 'C');
    }
    return std::move(make_pair(score_Int(f), std::move(f)));
}

//calculate the score by feature vector and weight vector
int CWS::score_Int(const fVec &f) const
{
    int ret = 0;
    for (auto &index : f)
    {
        ret += w[index];
    }
    return ret;
}

//make compare between stdf and myf
//if they are identical, return true; or return false
//and leave information for updating the weight vector in addf and minusf
//after  calling this function, all the fVec are sorted
bool CWS::cmp_f(fVec &stdf, fVec &myf, fVec &addf, fVec &minusf) const
{
    addf.clear();
    minusf.clear();
    sort(stdf.begin(), stdf.end());
    sort(myf.begin(), myf.end());
    auto it1 = stdf.begin(), it2 = myf.begin();
    while (it1 != stdf.end() && it2 != myf.end())
    {
        if (*it1 == *it2)
        {
            ++it1;
            ++it2;
        }
        else if (*it1 < *it2)
        {
            addf.push_back(*it1);
            ++it1;
        }
        else
        {
            minusf.push_back(*it2);
            ++it2;
        }
    }
    while (it1 != stdf.end())
    {
        addf.push_back(*(it1++));
    }
    while (it2 != myf.end())
    {
        minusf.push_back(*(it2++));
    }
    return addf.size() == 0;
}

void CWS::operator()(const string &_test_file, const string &_result_file) const
{
    ifstream is(_test_file);
    ofstream os(_result_file);
    string raw_line;         //a line of sentences scanned from a utf8 txt
    vector<string> std_line; //a line of characters
    vector<string> puresent;
    string mytags;

    while (getline(is, raw_line))
    {
        utf8_to_chrcs(raw_line, std_line);
        bool start_of_line = true;
        for (auto it1 = std_line.begin(), it2 = std_line.begin(); it1 != std_line.end();)
        {
            while (isPunc(*it1))
            {
                if (!start_of_line)
                {
                    os << "  ";
                }
                else
                {
                    start_of_line == false;
                }
                os << *it1;
                ++it1;
                if (it1 == std_line.end())
                    break;
            }
            if (it1 == std_line.end())
                break;
            it2 = it1 + 1;
            while (it2 != std_line.end() && !isPunc(*it2))
            {
                ++it2;
            }

            //it1~it2 is the range denotes a pure sentence waited to be segmented
            //get pure sentence
            puresent.clear();
            for (auto it = it1; it != it2; ++it)
            {
                puresent.push_back(std::move(*it));
            }
            //get mytags
            avr_segment(puresent, mytags);
            //output result
            auto it = puresent.begin();
            auto tit = mytags.begin();
            while (it != puresent.end())
            {
                if (start_of_line)
                {
                    start_of_line = false;
                }
                else if ((*tit == '1') || (*tit == '3'))
                {
                    os << "  ";
                }
                os << *it;
                ++it;
                ++tit;
            }

            if (it2 == std_line.end())
            {
                break;
            }
            else
            { //*it2 is a punctuation
                os << "  " << *it2;
                it1 = it2 + 1;
            }
        }
        os << endl;
    }

    is.close();
    os.close();
}

//segmentation with a pure sentence (at least one character)
//this function optimize mytags
//use avr_w
void CWS::avr_segment(const std::vector<std::string> &puresent, std::string &mytags) const
{
    //initialize mytags
    mytags.clear();
    size_t len = puresent.size();
    mytags.assign(len, '0');
    ++(mytags[0]);
    mytags[len - 1] += 2;

    if (len == 1)
    { //if there is only one character
        return;
    }

    //if there is at least two characters
    //construct the graph and make optimization
    size_t n = 2 * len - 1; //n>=3
    vector<Vertex_avr> graph;
    {
        graph.emplace_back(0);
        graph[0].scr = chrc_score_Double(puresent[0], string(), puresent[1], '3');
    }
    {
        graph.emplace_back(1);
        graph[1].scr = chrc_score_Double(puresent[0], string(), puresent[1], '1');
    }
    for (size_t i = 2; i < n - 1; ++i)
    {
        graph.emplace_back(i);
        size_t index = i / 2;
        char tag = (graph[i].cut ? '2' : '0');
        auto scr_c = chrc_score_Double(puresent[index], puresent[index - 1], puresent[index + 1], tag + 1) + graph[graph[i].pre_c].scr;
        auto scr_n = chrc_score_Double(puresent[index], puresent[index - 1], puresent[index + 1], tag) + graph[graph[i].pre_n].scr;
        if (scr_c > scr_n)
        {
            graph[i].pre = graph[i].pre_c;
            graph[i].scr = scr_c;
        }
        else
        {
            graph[i].pre = graph[i].pre_n;
            graph[i].scr = scr_n;
        }
    }
    {
        graph.emplace_back(n - 1);
        auto scr_c = chrc_score_Double(puresent[len - 1], puresent[len - 2], string(), '3') + graph[graph[n - 1].pre_c].scr;
        auto scr_n = chrc_score_Double(puresent[len - 1], puresent[len - 2], string(), '2') + graph[graph[n - 1].pre_n].scr;
        if (scr_c > scr_n)
        {
            graph[n - 1].pre = graph[n - 1].pre_c;
            graph[n - 1].scr = scr_c;
        }
        else
        {
            graph[n - 1].pre = graph[n - 1].pre_n;
            graph[n - 1].scr = scr_n;
        }
    }

    //back tracking
    size_t i = n - 1;
    while (i != -1)
    {
        size_t index = i / 2;
        if (index < len - 1 && graph[i].cut)
        {
            mytags[index] += 2;
            ++(mytags[index + 1]);
        }
        i = graph[i].pre;
    }
}

//returning the score of one character according to
//its left and right character(if none, assign them with empty string)
//and its tag
//use avr_w
double CWS::chrc_score_Double(const string &ch, const string &pre_ch, const string &post_ch, char tag) const
{
    fVec f;
    fVec_add(f, ch + '_' + tag + 'C');
    if (pre_ch.size())
    {
        fVec_add(f, pre_ch + '_' + tag + 'L');
        fVec_add(f, pre_ch + '_' + ch + '_' + tag + 'L');
    }
    if (post_ch.size())
    {
        fVec_add(f, post_ch + '_' + tag + 'R');
        fVec_add(f, ch + '_' + post_ch + '_' + tag + 'R');
    }
    if (pre_ch.size() && post_ch.size())
    {
        fVec_add(f, pre_ch + '_' + post_ch + '_' + tag + 'C');
    }
    return score_Double(f);
}

//calculate the score by feature vector and averaged weight vector
double CWS::score_Double(const fVec &f) const
{
    double ret = 0.0;
    for (auto &index : f)
    {
        ret += avr_w[index];
    }
    return ret;
}
