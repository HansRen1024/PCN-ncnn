//#include "PCN.h"
#include "net.h"

struct Window2{
    int x, y, w, h;
    float angle, scale, conf;
    Window2(int x_, int y_, int w_, int h_, float a_, float s_, float c_)
        : x(x_), y(y_), w(w_), h(h_), angle(a_), scale(s_), conf(c_)
    {}
};
class Impl{
public:
	void printNcnnMat(const ncnn::Mat& m);
	void printMat(cv::Mat& image);
    void LoadModel(const std::string model1,const std::string model2,const std::string model3,
    		const std::string net1, const std::string net2, const std::string net3);
    cv::Mat ResizeImg(cv::Mat img, float scale);
    static bool CompareWin(const Window2 &w1, const Window2 &w2);
    bool Legal(int x, int y, cv::Mat img);
    bool Inside(int x, int y, Window2 rect);
    int SmoothAngle(int a, int b);
    float IoU(Window2 &w1, Window2 &w2);
    std::vector<Window2> NMS(std::vector<Window2> &winList, bool local, float threshold);
    std::vector<Window2> DeleteFP(std::vector<Window2> &winList);
//    cv::Mat PreProcessImg(cv::Mat img);
    cv::Mat PreProcessImg(cv::Mat img,  int dim);
    cv::Mat PadImg(cv::Mat img);
    std::vector<Window> TransWindow(cv::Mat img, cv::Mat imgPad, std::vector<Window2> &winList);
    std::vector<Window2> Stage1(cv::Mat img, cv::Mat imgPad,
    		ncnn::Net &net_1, float thres);
    std::vector<Window2> Stage2(cv::Mat img, cv::Mat img180,
    		ncnn::Net &net_2, float thres, int dim, std::vector<Window2> &winList);
    std::vector<Window2> Stage3(cv::Mat img, cv::Mat img180, cv::Mat img90, cv::Mat imgNeg90,
    		ncnn::Net &net_3, float thres, int dim, std::vector<Window2> &winList);
    ncnn::Net net_1,net_2,net_3;
    int minFace_;
    float scale_;
    int stride_;
    float classThreshold_[3];
    float nmsThreshold_[3];
    float angleRange_;
    bool stable_;
    int threads = 4;
    const float mean_vals[3] = {104.0, 117.0, 123.0};
};
PCN::PCN(const std::string model1,const std::string model2,const std::string model3,
		const std::string net1, const std::string net2, const std::string net3) : impl_(new Impl()){
    Impl *p = (Impl *)impl_;
    p->LoadModel(model1, model2, model3, net1, net2, net3);
}
void PCN::SetVideoSmooth(bool stable){
    Impl *p = (Impl *)impl_;
    p->stable_ = stable;
}
void PCN::SetMinFaceSize(int minFace){
    Impl *p = (Impl *)impl_;
    p->minFace_ = minFace > 20 ? minFace : 20;
    p->minFace_ *= 1.4;
}
void PCN::SetScoreThresh(float thresh1, float thresh2, float thresh3){
    Impl *p = (Impl *)impl_;
    p->classThreshold_[0] = thresh1;
    p->classThreshold_[1] = thresh2;
    p->classThreshold_[2] = thresh3;
    p->nmsThreshold_[0] = 0.8;
    p->nmsThreshold_[1] = 0.8;
    p->nmsThreshold_[2] = 0.3;
    p->stride_ = 8;
    p->angleRange_ = 45;
}
void PCN::SetImagePyramidScaleFactor(float factor){
    Impl *p = (Impl *)impl_;
    p->scale_ = factor;
}
std::vector<Window> PCN::DetectFace(cv::Mat img){
    Impl *p = (Impl *)impl_;
    cv::Mat imgPad = p->PadImg(img);
    cv::Mat img180, img90, imgNeg90;
    cv::flip(imgPad, img180, 0);
    cv::transpose(imgPad, img90);
    cv::flip(img90, imgNeg90, 0);
    std::vector<Window2> winList = p->Stage1(img, imgPad, p->net_1, p->classThreshold_[0]);
    winList = p->NMS(winList, true, p->nmsThreshold_[0]);
    winList = p->Stage2(imgPad, img180, p->net_2, p->classThreshold_[1], 24, winList);
    winList = p->NMS(winList, true, p->nmsThreshold_[1]);
    winList = p->Stage3(imgPad, img180, img90, imgNeg90, p->net_3, p->classThreshold_[2], 48, winList);
    winList = p->NMS(winList, false, p->nmsThreshold_[2]);
    winList = p->DeleteFP(winList);
    static std::vector<Window2> preList;
    if (p->stable_){
        for (uint i = 0; i < winList.size(); i++){
            for (uint j = 0; j < preList.size(); j++){
                if (p->IoU(winList[i], preList[j]) > 0.9)
                    winList[i] = preList[j];
                else if (p->IoU(winList[i], preList[j]) > 0.6){
                    winList[i].x = (winList[i].x + preList[j].x) / 2;
                    winList[i].y = (winList[i].y + preList[j].y) / 2;
                    winList[i].w = (winList[i].w + preList[j].w) / 2;
                    winList[i].h = (winList[i].h + preList[j].h) / 2;
                    winList[i].angle = p->SmoothAngle(winList[i].angle, preList[j].angle);
                }
            }
        }
        preList = winList;
    }
    return p->TransWindow(img, imgPad, winList);
}
void Impl::printNcnnMat(const ncnn::Mat& m){
    for (int q=0; q<m.c; q++){
        const float* ptr = m.channel(q);
        for (int y=0; y<m.h; y++){
            for (int x=0; x<m.w; x++)
                printf("%f ", ptr[x]);
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
    }
}
void Impl::printMat(cv::Mat& image){
	int nr=image.rows;
    int nl=image.cols*image.channels();
    for(int k=0;k<nr;k++){
    	const uchar* inData=image.ptr<uchar>(k);
        for(int i=0;i<nl;i++)
        	printf("%d ", inData[i]);
        printf("\n");
     }
}
void Impl::LoadModel(const std::string model1,const std::string model2,const std::string model3,
		const std::string net1, const std::string net2, const std::string net3){
	net_1.load_param(net1.c_str());
	net_1.load_model(model1.c_str());
	net_2.load_param(net2.c_str());
	net_2.load_model(model2.c_str());
	net_3.load_param(net3.c_str());
	net_3.load_model(model3.c_str());
}
//cv::Mat Impl::PreProcessImg(cv::Mat img){
//    cv::Mat mean(img.size(), CV_32FC3, cv::Scalar(104, 117, 123));
//    cv::Mat imgF;
//    img.convertTo(imgF, CV_32FC3);
//    return imgF - mean;
//}
cv::Mat Impl::PreProcessImg(cv::Mat img, int dim){
    cv::Mat imgNew;
    cv::resize(img, imgNew, cv::Size(dim, dim));
    cv::Mat mean(imgNew.size(), CV_32FC3, cv::Scalar(104, 117, 123));
    cv::Mat imgF;
    imgNew.convertTo(imgF, CV_32FC3);
    return imgF - mean;
}
cv::Mat Impl::ResizeImg(cv::Mat img, float scale){
    cv::Mat ret;
    cv::resize(img, ret, cv::Size(int(img.cols / scale), int(img.rows / scale)));
    return ret;
}
bool Impl::CompareWin(const Window2 &w1, const Window2 &w2){
    return w1.conf > w2.conf;
}
bool Impl::Legal(int x, int y, cv::Mat img){
    if (x >= 0 && x < img.cols && y >= 0 && y < img.rows) return true;
    else return false;
}
bool Impl::Inside(int x, int y, Window2 rect){
    if (x >= rect.x && y >= rect.y && x < rect.x + rect.w && y < rect.y + rect.h) return true;
    else return false;
}
int Impl::SmoothAngle(int a, int b){
    if (a > b) std::swap(a, b);
    int diff = (b - a) % 360;
    if (diff < 180) return a + diff / 2;
    else return b + (360 - diff) / 2;
}
float Impl::IoU(Window2 &w1, Window2 &w2){
    int xOverlap = std::max(0, std::min(w1.x + w1.w - 1, w2.x + w2.w - 1) - std::max(w1.x, w2.x) + 1);
    int yOverlap = std::max(0, std::min(w1.y + w1.h - 1, w2.y + w2.h - 1) - std::max(w1.y, w2.y) + 1);
    int intersection = xOverlap * yOverlap;
    int unio = w1.w * w1.h + w2.w * w2.h - intersection;
    return float(intersection) / unio;
}
std::vector<Window2> Impl::NMS(std::vector<Window2> &winList, bool local, float threshold){
    if (winList.size() == 0) return winList;
    std::sort(winList.begin(), winList.end(), CompareWin);
    bool flag[winList.size()];
    memset(flag, 0, winList.size());
    for (uint i = 0; i < winList.size(); i++){
        if (flag[i]) continue;
        for (uint j = i + 1; j < winList.size(); j++){
            if (local && abs(winList[i].scale - winList[j].scale) > EPS) continue;
            if (IoU(winList[i], winList[j]) > threshold) flag[j] = 1;
        }
    }
    std::vector<Window2> ret;
    for (uint i = 0; i < winList.size(); i++){
        if (!flag[i]) ret.push_back(winList[i]);
    }
    return ret;
}
/// to delete some false positives
std::vector<Window2> Impl::DeleteFP(std::vector<Window2> &winList){
    if (winList.size() == 0) return winList;
    std::sort(winList.begin(), winList.end(), CompareWin);
    bool flag[winList.size()];
    memset(flag, 0, winList.size());
    for (uint i = 0; i < winList.size(); i++){
        if (flag[i]) continue;
        for (uint j = i + 1; j < winList.size(); j++){
            if (Inside(winList[j].x, winList[j].y, winList[i]) && Inside(winList[j].x + winList[j].w - 1, winList[j].y + winList[j].h - 1, winList[i]))
                flag[j] = 1;
        }
    }
    std::vector<Window2> ret;
    for (uint i = 0; i < winList.size(); i++){
        if (!flag[i]) ret.push_back(winList[i]);
    }
    return ret;
}
/// to detect faces on the boundary
cv::Mat Impl::PadImg(cv::Mat img){
    int row = std::min(int(img.rows * 0.2), 100);
    int col = std::min(int(img.cols * 0.2), 100);
    cv::Mat ret;
    cv::copyMakeBorder(img, ret, row, row, col, col, cv::BORDER_CONSTANT);
    return ret;
}
std::vector<Window2> Impl::Stage1(cv::Mat img, cv::Mat imgPad, ncnn::Net &net_1, float thres){
    int row = (imgPad.rows - img.rows) / 2;
    int col = (imgPad.cols - img.cols) / 2;
    std::vector<Window2> winList;
    int netSize = 24;
    float curScale;
    curScale = minFace_ / float(netSize);
    cv::Mat imgResized = ResizeImg(img, curScale);
    while (std::min(imgResized.rows, imgResized.cols) >= netSize){
    	ncnn::Extractor ex_1 = net_1.create_extractor();
		ex_1.set_light_mode(true);
		ex_1.set_num_threads(threads);
    	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(imgResized.data, ncnn::Mat::PIXEL_BGR, imgResized.cols, imgResized.rows);
    	ncnn_img.substract_mean_normalize(mean_vals, 0);
    	ex_1.input("data", ncnn_img);
    	ncnn::Mat reg,prob,rotateProb;
        ex_1.extract("bbox_reg_1", reg);
        ex_1.extract("rotate_cls_prob", rotateProb);
        ex_1.extract("cls_prob", prob);
        float w = netSize * curScale;
        const float* ptrProb = prob.channel(1);
        const float* ptrRotate = rotateProb.channel(1);
        const float* ptrReg_1 = reg.channel(0);
        const float* ptrReg_2 = reg.channel(1);
        const float* ptrReg_3 = reg.channel(2);
        for (int i = 0; i < prob.h; i++){
            for (int j = 0; j < prob.w; j++){
                if (ptrProb[j] > thres){
                    float sn = ptrReg_1[j];
                    float xn = ptrReg_2[j];
                    float yn = ptrReg_3[j];
                    int rx = int(j * curScale * stride_ - 0.5 * sn * w + sn * xn * w + 0.5 * w) + col;
                    int ry = int(i * curScale * stride_ - 0.5 * sn * w + sn * yn * w + 0.5 * w) + row;
                    int rw = int(w * sn);
                    if (Legal(rx, ry, imgPad) && Legal(rx + rw - 1, ry + rw - 1, imgPad)){
                        if (ptrRotate[j] > 0.5)
                            winList.push_back(Window2(rx, ry, rw, rw, 0, curScale, ptrProb[j]));
                        else
                            winList.push_back(Window2(rx, ry, rw, rw, 180, curScale, ptrProb[j]));
                    }
                }
            }
            ptrProb += prob.w;
            ptrReg_1 += prob.w;
            ptrReg_2 += prob.w;
            ptrReg_3 += prob.w;
            ptrRotate += prob.w;
        }
        imgResized = ResizeImg(imgResized, scale_);
        curScale = float(img.rows) / imgResized.rows;
    }
    return winList;
}
std::vector<Window2> Impl::Stage2(cv::Mat img, cv::Mat img180, ncnn::Net &net_2, float thres, int dim, std::vector<Window2> &winList){
    if (winList.size() == 0)
        return winList;
    std::vector<cv::Mat> dataList;
    int height = img.rows;
    for (uint i = 0; i < winList.size(); i++){
    	if (abs(winList[i].angle) < EPS)
    		dataList.push_back(img(cv::Rect(winList[i].x, winList[i].y, winList[i].w, winList[i].h)).clone());
    	else{
			int y2 = winList[i].y + winList[i].h - 1;
			dataList.push_back(img180(cv::Rect(winList[i].x, height - 1 - y2, winList[i].w, winList[i].h)).clone());
		}
	}
    std::vector<Window2> ret;
//    printMat(dataList[0]);
    for(uint ind=0; ind< winList.size(); ind++){
    	ncnn::Extractor ex_2 = net_2.create_extractor();
		ex_2.set_light_mode(true);
		ex_2.set_num_threads(threads);
    	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels_resize(dataList[ind].data, ncnn::Mat::PIXEL_BGR, dataList[ind].cols, dataList[ind].rows,dim,dim);
    	ncnn_img.substract_mean_normalize(mean_vals, 0);
//    	printNcnnMat(ncnn_img);
    	ex_2.input("data", ncnn_img);
    	ncnn::Mat reg,prob,rotateProb;
    	ex_2.extract("bbox_reg_2", reg);
    	ex_2.extract("rotate_cls_prob", rotateProb);
    	ex_2.extract("cls_prob", prob);
		if (prob[1] > thres){
			float sn = reg[0];
			float xn = reg[1];
			float yn = reg[2];
			int cropX = winList[ind].x;
			int cropY = winList[ind].y;
			int cropW = winList[ind].w;
			if (abs(winList[ind].angle)  > EPS)
				cropY = height - 1 - (cropY + cropW - 1);
			int w = int(sn * cropW);
			int x = int(cropX  - 0.5 * sn * cropW + cropW * sn * xn + 0.5 * cropW);
			int y = int(cropY  - 0.5 * sn * cropW + cropW * sn * yn + 0.5 * cropW);
			float maxRotateScore = 0;
			int maxRotateIndex = 0;
			for (int j = 0; j < 3; j++){
				if (rotateProb[j] > maxRotateScore){
					maxRotateScore = rotateProb[j];
					maxRotateIndex = j;
				}
			}
			if (Legal(x, y, img) && Legal(x + w - 1, y + w - 1, img)){
				float angle = 0;
				if (abs(winList[ind].angle)  < EPS){
					if (maxRotateIndex == 0)
						angle = 90;
					else if (maxRotateIndex == 1)
						angle = 0;
					else
						angle = -90;
					ret.push_back(Window2(x, y, w, w, angle, winList[ind].scale, prob[1]));
				}
				else{
					if (maxRotateIndex == 0)
						angle = 90;
					else if (maxRotateIndex == 1)
						angle = 180;
					else
						angle = -90;
					ret.push_back(Window2(x, height - 1 -  (y + w - 1), w, w, angle, winList[ind].scale, prob[1]));
				}
			}
		}
    }
    return ret;
}
std::vector<Window2> Impl::Stage3(cv::Mat img, cv::Mat img180, cv::Mat img90, cv::Mat imgNeg90, ncnn::Net &net_3, float thres, int dim, std::vector<Window2> &winList){
    if (winList.size() == 0)
        return winList;
    std::vector<cv::Mat> dataList;
    int height = img.rows;
    int width = img.cols;
    for (uint i = 0; i < winList.size(); i++){
        if (abs(winList[i].angle) < EPS)
            dataList.push_back(img(cv::Rect(winList[i].x, winList[i].y, winList[i].w, winList[i].h)).clone());
        else if (abs(winList[i].angle - 90) < EPS)
            dataList.push_back(img90(cv::Rect(winList[i].y, winList[i].x, winList[i].h, winList[i].w)).clone());
        else if (abs(winList[i].angle + 90) < EPS){
            int x = winList[i].y;
            int y = width - 1 - (winList[i].x + winList[i].w - 1);
            dataList.push_back(imgNeg90(cv::Rect(x, y, winList[i].w, winList[i].h)).clone());
        }
        else{
            int y2 = winList[i].y + winList[i].h - 1;
            dataList.push_back(img180(cv::Rect(winList[i].x, height - 1 - y2, winList[i].w, winList[i].h)).clone());
        }
    }
    std::vector<Window2> ret;
    for(uint ind=0; ind< winList.size(); ind++){
    	ncnn::Extractor ex_3 = net_3.create_extractor();
    	ex_3.set_light_mode(true);
    	ex_3.set_num_threads(threads);
    	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels_resize(dataList[ind].data, ncnn::Mat::PIXEL_BGR, dataList[ind].cols, dataList[ind].rows,dim,dim);
    	ncnn_img.substract_mean_normalize(mean_vals, 0);
    	ex_3.input("data", ncnn_img);
    	ncnn::Mat reg,prob,rotateProb;
    	ex_3.extract("bbox_reg_3", reg);
    	ex_3.extract("rotate_reg_3", rotateProb);
    	ex_3.extract("cls_prob", prob);
        if (prob[1] > thres){
        	float sn = reg[0];
			float xn = reg[1];
			float yn = reg[2];
            int cropX = winList[ind].x;
            int cropY = winList[ind].y;
            int cropW = winList[ind].w;
            cv::Mat imgTmp = img;
            if (abs(winList[ind].angle - 180)  < EPS){
                cropY = height - 1 - (cropY + cropW - 1);
                imgTmp = img180;
            }
            else if (abs(winList[ind].angle - 90)  < EPS){
                std::swap(cropX, cropY);
                imgTmp = img90;
            }
            else if (abs(winList[ind].angle + 90)  < EPS){
                cropX = winList[ind].y;
                cropY = width - 1 - (winList[ind].x + winList[ind].w - 1);
                imgTmp = imgNeg90;
            }
            int w = int(sn * cropW);
            int x = int(cropX  - 0.5 * sn * cropW + cropW * sn * xn + 0.5 * cropW);
            int y = int(cropY  - 0.5 * sn * cropW + cropW * sn * yn + 0.5 * cropW);
            float angle = angleRange_ * rotateProb[0];
            if (Legal(x, y, imgTmp) && Legal(x + w - 1, y + w - 1, imgTmp)){
                if (abs(winList[ind].angle)  < EPS)
                    ret.push_back(Window2(x, y, w, w, angle, winList[ind].scale, prob[1]));
                else if (abs(winList[ind].angle - 180)  < EPS)
                    ret.push_back(Window2(x, height - 1 -  (y + w - 1), w, w, 180 - angle, winList[ind].scale, prob[1]));
                else if (abs(winList[ind].angle - 90)  < EPS)
                    ret.push_back(Window2(y, x, w, w, 90 - angle, winList[ind].scale, prob[1]));
                else
                    ret.push_back(Window2(width - y - w, x, w, w, -90 + angle, winList[ind].scale, prob[1]));
            }
        }
    }
    return ret;
}
std::vector<Window> Impl::TransWindow(cv::Mat img, cv::Mat imgPad, std::vector<Window2> &winList){
    int row = (imgPad.rows - img.rows) / 2;
    int col = (imgPad.cols - img.cols) / 2;
    std::vector<Window> ret;
    for(uint i = 0; i < winList.size(); i++){
        if (winList[i].w > 0 && winList[i].h > 0)
            ret.push_back(Window(winList[i].x - col, winList[i].y - row, winList[i].w, winList[i].angle, winList[i].conf));
    }
    return ret;
}
int main()
{
    PCN detector("model/PCN-1.bin", "model/PCN-2.bin", "model/PCN-3.bin",
    		"model/PCN-1.proto", "model/PCN-2.proto", "model/PCN-3.proto");
    detector.SetMinFaceSize(45);
    detector.SetScoreThresh(0.37, 0.43, 0.95);
    detector.SetImagePyramidScaleFactor(1.414);
    detector.SetVideoSmooth(true);
    cv::VideoCapture capture(0);
    cv::Mat img;
    cv::TickMeter tm;
    while (1)
    {
        capture >> img;
        if (img.empty()) {
			break;
		}
        tm.reset();
        tm.start();
        std::vector<Window> faces = detector.DetectFace(img);
        tm.stop();
        int fps = 1000.0 / tm.getTimeMilli();
        std::stringstream ss;
        ss << fps;
        cv::putText(img, ss.str() + "FPS",
                    cv::Point(20, 45), 4, 1, cv::Scalar(0, 0, 125));
        for (uint i = 0; i < faces.size(); i++)
        {
            DrawFace(img, faces[i]);
        }
        cv::imshow("PCN", img);
        if (cv::waitKey(1) == 'q')
            break;
    }

    capture.release();
    cv::destroyAllWindows();

    return 0;
}
