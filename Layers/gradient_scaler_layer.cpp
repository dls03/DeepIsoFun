#include <algorithm>
#include <vector>
#include <stdio.h>
#include "caffe/layer.hpp"
#include "caffe/messenger.hpp"
#include "caffe/layers/gradient_scaler_layer.hpp"

namespace caffe {

class AdaptationCoefficientHandler: public Listener {

 public:
  AdaptationCoefficientHandler(float lower_bound, float upper_bound, 
                               float alpha, float max_iter, float* coeff)
      : lower_bound_(lower_bound), upper_bound_(upper_bound), alpha_(alpha),
        max_iter_(max_iter), coeff_(*coeff) {
    height_ = upper_bound_ - lower_bound_;
	//printf("height_ %f \n", height_);
    }

  void handle(void* message) {
    int iter = *(static_cast<int*>(message));
    float progress = std::min(1.f, static_cast<float>(iter) / max_iter_);

    coeff_ = 2.f * height_ / (1.f + exp(-alpha_ * progress)) - 
             height_ + lower_bound_;

     LOG(INFO) << "iter = " << iter << " progress = " << progress << " coeff = " << coeff_;
     printf("iter %d, progress %f, coeff %f \n\n\n",iter, progress, coeff_);
     //cout << "iter = " << iter << " progress = " << progress << " coeff = " << coeff_;
  }

 private:
  float lower_bound_, upper_bound_, alpha_, max_iter_, height_;
  float& coeff_;
};


template <typename Dtype>
void GradientScalerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
	printf("entered\n\n\n");
  lower_bound_ = this->layer_param_.gradient_scaler_param().lower_bound();
  upper_bound_ = this->layer_param_.gradient_scaler_param().upper_bound();
  alpha_ = this->layer_param_.gradient_scaler_param().alpha();
  max_iter_ = this->layer_param_.gradient_scaler_param().max_iter();
  coeff_ = 1.f; // Default adaptation coefficient.

  DCHECK(lower_bound_ <= upper_bound_);
  DCHECK(alpha_ >= 0.f);
  DCHECK(max_iter_ >= 1.f);
  
  Messenger::AddListener("SOLVER_ITER_CHANGED", 
      new AdaptationCoefficientHandler(lower_bound_, upper_bound_, 
                                       alpha_, max_iter_, &coeff_));

}

template <typename Dtype>
void GradientScalerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ShareData(*bottom[0]);
  //printf("grlforwardcpu\n");
}

template <typename Dtype>
void GradientScalerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for(int i=0;i<2;i++){
  	if (propagate_down[i]) {
    		const int count = bottom[i]->count();
		//printf("grlbackwardcpu %d\n",count);
    		const Dtype* top_diff = top[i]->cpu_diff();
    		Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();

    		caffe_cpu_scale(count, Dtype(-coeff_), top_diff, bottom_diff);
  	}
  }
}

#ifdef CPU_ONLY
STUB_GPU(GradientScalerLayer);
#endif

INSTANTIATE_CLASS(GradientScalerLayer);
REGISTER_LAYER_CLASS(GradientScaler);

}  // namespace caffe
