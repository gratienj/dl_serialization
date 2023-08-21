#include "TFcaller.h"
#include <chrono>

int main() {
    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    //
    TFcaller net("../../model/wall_functions/AI_wall_function_TF");
    //
    double* data = new double[12];
    double* result;
    //
    // Test data are hardcoded for now
    //
    double inputs[31][12] = { {0.0004330561721637364,1.2472816018540796e-05,0.5568898501769738,0.6513838235178302,-0.46030552821005544,-1.369933638471612,-0.009292382842992305,-0.056488687501122486,0.0005029013539648814,0.00018768783308831908,0.047932916960537725,0.40499197481472915},
                            {0.005626602501775812,0.0001620565236686582,0.8830390211536039,0.4500554764567432,-0.5848221064273114,-2.1194109429565993,-0.004647859386513842,-0.09901465032147715,0.009527594220174395,0.002179869966575412,0.025604542527667338,0.21064291071857863},
                            {0.02317987897429049,0.0006676232423470763,1.1893619079589843,-0.054844708658033835,0.012739066216253467,-2.4074874492592526,-0.05639472442473358,-1.2105957666371294,0.18050269880678413,0.0011823502910724753,0.00010709651763082378,0.0004155779865113161},
                            {0.002556322314154168,7.362679476250485e-05,0.68248879273586,0.6172739145305299,-0.46046875717233404,-1.6207296676987328,-0.008328320003947707,-0.0686732546730125,0.003304327549043141,0.0009079114482043774,0.04164493385565588,0.345236534373511},
                            {0.004348251110260388,0.000125237647184919,1.564779287311888,0.06568437959521597,-0.007176546845018589,-3.1222680202782978,0.006341199879536026,-0.5735378525343133,0.06260132689960306,0.0018137317767032164,-0.0011337208222309,-0.026522582934579125},
                            {0.0018194367173492633,5.240313126005943e-05,0.5920307918144261,0.7247065642044841,-0.3443689404325002,-1.4048262466922885,-0.00867211345645145,-0.042504439475154636,0.001145995547149929,0.00048693857370152717,0.04543377136393786,0.397717712372532},
                            {0.01392906814950952,0.0004011828384075323,1.5101651304746004,0.3045564513316353,-0.21811760915589223,-3.1156321267686136,-0.020638649434641727,-0.17588678337423966,0.021711177480985437,0.004568790196587649,-0.0024851267851556708,-0.05211753440162999},
                            {0.021198893197670512,0.000610567200393736,0.613323991766723,-0.07378202663368894,0.02129069835029989,-1.2291292091756374,-0.07949856752021646,-1.1197820762487325,0.41132378636495737,0.0011578898470034028,0.0007043347449030176,-6.222469292818574e-05},
                            {0.009453007393443642,0.000272264037829598,1.0744265426424793,0.6971085092570926,-0.06606454584675452,-2.1646609196113555,-0.02012909397467999,-0.08688609661888178,0.007529796107466533,0.0027302185363795254,0.03549339929392866,0.24365817260806996},
                            {0.019702428634265603,0.0005674662625076498,1.3252100661317325,-0.03127366262427124,-0.023189600579749607,-2.644850716445853,-0.03831547045677831,-1.0825258735193846,0.14265390480049603,0.0012328979133048284,-0.0006098646956252217,-7.579465552928098e-05},
                            {0.004574385244176135,0.00013175072707880575,0.5976240557833202,1.9722742478331112,0.13400895130139892,-1.1914071777625017,-0.03597633223126655,0.06178987192741539,0.002611458059779489,0.0011137153635047505,0.012981741367971889,0.424453769699959},
                            {0.03288576748513956,0.0009471707224982591,1.5832134720059583,-0.03045578873102566,0.03454382219182731,-3.1640738294750865,-0.05453060337037406,-0.4635359763471291,0.049474737978737734,0.0023177816152835915,0.0006693484457156504,-0.037235881407902044},
                            {0.0034326052267430567,9.886535791310647e-05,0.4692787223147907,4.1599287115000365,0.18345477569158175,-0.8728304855416997,-0.023736187166633907,-0.08024624304822156,0.0009056969273338124,0.00040392460230594807,0.0013980076804599845,0.49633070247774663},
                            {0.05700543248158996,0.0016418615346080058,1.631094016079529,0.8750336039319018,0.1192412199204968,-3.202184338547667,-0.08825011296798745,-0.1484131106618646,0.017158658933929052,0.004487655622241335,-0.008647847863679968,0.004547819179577119},
                            {0.04259603364127023,0.0012268442869029445,0.8068004804338729,-0.12447397145917338,0.011828787553145589,-1.6209379027757778,-0.1344720475380874,-1.2866927325802768,0.3250751631425488,0.0012399238223580339,0.002287375030541071,-0.0003306642812993867},
                            {0.062219173677506,0.0017920268916332372,1.0873865587085074,8.020388625747048,0.21628595480501378,-2.1014385980702768,-0.1787630628766347,-0.16615833186149376,0.005950907239516372,0.0025298452429343182,-0.005712607944380539,0.326785009041846},
                            {0.1305457496423326,0.0037599582270257084,1.3779608680268038,-0.2825160153648271,-0.00376686559514516,-2.7741306720505117,-0.27242103464379785,-0.9377210622614641,0.11274145311595848,0.0013221849471481905,0.001512243259168025,-0.0028311436674868236},
                            {0.05343766874432017,0.0015391033624516178,0.8030612648115729,25.35169695060484,0.2676632522372057,-1.442801228466122,-0.390967381510631,-0.37951369504007704,0.002063873237979164,0.0010078900311862992,0.00296060440051163,0.499268029709201},
                            {0.28946778741300383,0.008337205858669466,1.5509194397816461,-0.5256938144666259,-0.025153464936990343,-3.1541890615999426,-0.5100619852328959,-0.38957527885039256,0.03910060408128902,0.0027197641245271607,0.0011552227027210313,-0.04165666929314381},
                            {0.0520108035907218,0.001498007015861803,0.5129780238279,74.6256396439372,0.22898031482608638,-0.9156433701756914,-0.7529856082448263,-0.18941870609212078,0.000715785437581344,0.00041588183957045054,0.0011004270939481762,0.6263842112133656},
                            {0.5712780534946424,0.016453860987748918,1.552336659796227,-0.08622926200479956,-0.04503265764965989,-3.1853245680848574,-0.9347255017955942,-0.12728329212363948,0.013560737397533574,0.004380371470387458,0.003962179437745194,-0.04865399718945348},
                            {0.4168637886574741,0.012006445525848909,0.1837446979575443,-1.790680900327621,0.011292641532036565,-1.0509787930958832,-1.1494420249462787,-1.147291482991317,0.2569116233856535,0.017082890680660604,0.10770761072300217,0.08825300285088063},
                            {0.738688218428792,0.0212755823280182,0.2407257292787051,96.13131615423389,-0.05324175207955497,-1.3139053153464322,-0.9738010742768184,-0.04375946116859083,0.004703088432663193,0.005915776262513396,0.012195256597417698,0.9147941218814042},
                            {0.5094647956883304,0.014673525221438086,-0.6765939369025867,-4.913802800754248,-0.020138784474491526,0.13431416037774857,-0.09681157950963701,-0.15244862689861347,0.1542944634322175,0.018296676001147462,0.00516198944733075,-0.06350938994297825},
                            {0.8374311956941807,0.024119562087966034,-0.40919555910172006,47.977250072004615,-0.055802347350230416,-0.21828663656349007,-0.013338754987652881,-0.12214650433984342,0.010305045908540561,0.020789200502824297,0.023198978877973363,-0.46517049411150796},
                            {0.13075594528479512,0.0037660122489860345,0.07855170233238679,-2.2140411343200967,-0.008854032872459306,-0.1495168842192619,-0.00796651137748799,0.10090630902804336,0.19523171902941847,0.018195738980059703,0.01620467384105231,-0.2797594441097514},
                            {0.20452030518087927,0.005890561785163574,-0.25012897171512727,51.22055211333526,-0.14059951151131486,0.12707360171832247,-0.01952853023888172,-0.052430450083655586,0.003573960677045757,0.008640583361622208,-0.005566556299430713,2.5777447152931154},
                            {0.405308782779676,0.01167364005701832,-0.11239084755330593,-0.4039769946682838,0.004309559292507611,-0.002194369031536964,0.0036718328511151158,-0.05500320392930983,0.06770959517461021,0.014163753234430525,-0.011078316859206194,0.021023965059613473},
                            {0.01574252401628802,0.0004534137101465444,-0.013947357021718532,13.27758502844182,-0.0010889211063560805,0.007035723239046089,-0.005972506316495126,0.0015589991561438252,0.0012395087837961157,0.00027983537253001333,-0.0009498187135876941,0.23849784693752082},
                            {0.17087857844093426,0.004921618042653637,-0.011754222318324075,4.521325922671551,0.004098073666546202,-0.012560263020651682,-0.003201047107224486,-0.0053203613122307656,0.023482809563435025,0.001849965721140315,-0.0008431279149665389,0.05959901820021205},
                            {0.20061146279084519,0.005777979919091163,0.018432235216764813,-0.1805477732996787,9.343222431701176e-06,-0.03957647056403798,-0.005767359184027453,0.00017923060574786027,0.44488780733233735,0.0007525176985776367,-0.00018300623513014378,-0.0025468744731693177}};
    //
    size_t j = 0;
    for (size_t i = 0; i < 1000000; i++) { // i < 31 to check results
        for (size_t k = 0; k < 12; k++) {
            data[k] = inputs[j][k];
        }
        j++;
        if (j == 31) j = 0; // comment to check results
        result = net.run_ai(data);
        // cout << "result : " << result[0] << endl; // uncomment to check results
    }
    end = std::chrono::steady_clock::now();
    cout << "time : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << endl << endl;
    //
    delete[] data;
    //
    return 0;
}