#include <cmath>
#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <array>
#include <cstddef>
#include "mgard/compress.hpp"
#include <ctime>
using namespace std;


// dims example cesm 2 1800 3600 for the hirarachy when using python interface 
// cli is 2 3600 1800 


class Timer{
public:
    void start(){
        err = clock_gettime(CLOCK_REALTIME, &start_time);
    }
    void end(){
        err = clock_gettime(CLOCK_REALTIME, &end_time);
    }
    double get(){
        return (double)(end_time.tv_sec - start_time.tv_sec) + (double)
(end_time.tv_nsec - start_time.tv_nsec)/(double)1000000000;
    }
private:
    int err = 0;
    struct timespec start_time, end_time;
};
// copy from cpSZ
template<typename Type>
Type * readfile(const char * file, size_t& num){
	std::ifstream fin(file, std::ios::binary);
	if(!fin){
        std::cout << " Error, Couldn't find the file" << "\n";
        return 0;
    }
    fin.seekg(0, std::ios::end);
    const size_t num_elements = fin.tellg() / sizeof(Type);
    fin.seekg(0, std::ios::beg);
    Type * data = (Type *) malloc(num_elements*sizeof(Type));
	fin.read(reinterpret_cast<char*>(&data[0]), num_elements*sizeof(Type));
	fin.close();
	num = num_elements;
	return data;
}

template<typename Type>
void verify(const Type * ori_data, const Type * data, size_t num_elements){
    size_t i = 0;
    Type Max = 0, Min = 0, diffMax = 0;
    Max = ori_data[0];
    Min = ori_data[0];
    diffMax = fabs(data[0] - ori_data[0]);
    size_t k = 0;
    double sum1 = 0, sum2 = 0;
    for (i = 0; i < num_elements; i++){
        sum1 += ori_data[i];
        sum2 += data[i];
    }
    double mean1 = sum1/num_elements;
    double mean2 = sum2/num_elements;

    double sum3 = 0, sum4 = 0;
    double sum = 0, prodSum = 0, relerr = 0;

    double maxpw_relerr = 0; 
    for (i = 0; i < num_elements; i++){
        if (Max < ori_data[i]) Max = ori_data[i];
        if (Min > ori_data[i]) Min = ori_data[i];
        
        Type err = fabs(data[i] - ori_data[i]);
        if(ori_data[i]!=0 && fabs(ori_data[i])>1)
        {
            relerr = err/fabs(ori_data[i]);
            if(maxpw_relerr<relerr)
                maxpw_relerr = relerr;
        }

        if (diffMax < err)
            diffMax = err;
        prodSum += (ori_data[i]-mean1)*(data[i]-mean2);
        sum3 += (ori_data[i] - mean1)*(ori_data[i]-mean1);
        sum4 += (data[i] - mean2)*(data[i]-mean2);
        sum += err*err; 
    }
    double std1 = sqrt(sum3/num_elements);
    double std2 = sqrt(sum4/num_elements);
    double ee = prodSum/num_elements;
    double acEff = ee/std1/std2;

    double mse = sum/num_elements;
    double range = Max - Min;
    double psnr = 20*log10(range)-10*log10(mse);
    double nrmse = sqrt(mse)/range;

    printf ("Min=%.20G, Max=%.20G, range=%.20G\n", Min, Max, range);
    printf ("Max absolute error = %.10f\n", diffMax);
    printf ("Max relative error = %f\n", diffMax/(Max-Min));
    printf ("Max pw relative error = %f\n", maxpw_relerr);
    printf ("PSNR = %f, NRMSE= %.20G\n", psnr,nrmse);
    printf ("acEff=%f\n", acEff);   
}



template<typename Type>
void writefile(const char * file, Type * data, size_t num_elements){
	std::ofstream fout(file, std::ios::binary);
	fout.write(reinterpret_cast<const char*>(&data[0]), num_elements*sizeof(Type));
	fout.close();
}

template<typename Type>
mgard::DecompressedDataset<2, Type> compress_decompress_2d(Type *data,size_t r1, size_t r2, float eb, float s, size_t& compressed_size){
    Type max = data[0];
    Type min = data[0];
    for(size_t i=0; i<r1*r2; i++){
        if(data[i]>max) max = data[i];
        if(data[i]<min) min = data[i];
    }
    Timer timer; 
    eb = (max-min)*eb;
    const array<size_t, 2> dims = {r1, r2};
    const mgard::TensorMeshHierarchy<2, Type> hierarchy(dims);
    const size_t ndof = hierarchy.ndof();
    timer.start();
    mgard::CompressedDataset<2, Type> compressed = mgard::compress(hierarchy, data, s, eb);
    timer.end();
    printf("mgard compress time = %f \n", timer.get());
    timer.start();
    mgard::DecompressedDataset<2, Type> decompressed = mgard::decompress(compressed);
    timer.end();
    printf("mgard decompress time = %f \n", timer.get());
    compressed_size = compressed.size();
    std::cout<< "compressed size = " << compressed.size() << std::endl;
    verify(data, decompressed.data(), r1*r2);
    return decompressed;
}

template<typename Type>
void compress_decompress_2d(Type *data, Type *ddata, size_t r1, size_t r2, float eb, float s, size_t& compressed_size)
{
    mgard::DecompressedDataset<2, Type> decompressed = compress_decompress_2d(data,  r1,  r2, eb,  s,  compressed_size);
    std::copy(decompressed.data(), decompressed.data()+r1*r2,ddata);
}


template<typename Type>
mgard::DecompressedDataset<3, Type> compress_decompress_3d(Type *data, size_t r1, size_t r2, size_t r3, float eb, float s, size_t& compressed_size){
    Type max = data[0];
    Type min = data[0];
    for(size_t i=0; i<r1*r2*r3; i++){
        if(data[i]>max) max = data[i];
        if(data[i]<min) min = data[i];
    }
    Timer timer; 
    eb = (max-min)*eb;
    const array<size_t,3> dims = {r1, r2,r3};
    const mgard::TensorMeshHierarchy<3, Type> hierarchy(dims);
    const size_t ndof = hierarchy.ndof();
    timer.start();
    mgard::CompressedDataset<3, Type> compressed = mgard::compress(hierarchy, data, s, eb);
    timer.end();
    printf("mgard compress time = %f \n", timer.get());
    timer.start();
    mgard::DecompressedDataset<3, Type> decompressed = mgard::decompress(compressed);
    timer.end();
    printf("mgard decompress time = %f \n", timer.get());
    compressed_size = compressed.size();
    verify(data, decompressed.data(), r1*r2*r3);
    return decompressed;
}


template<typename Type>
void compress_decompress_3d(Type *data, Type *ddata, size_t r1, size_t r2, size_t r3, float eb, float s, size_t& compressed_size){
    mgard::DecompressedDataset<3, Type> decompressed = compress_decompress_3d(data,  r1,  r2,  r3,  eb,  s,  compressed_size);
    std::copy(decompressed.data(), decompressed.data()+r1*r2*r3,ddata);
}


template<typename Type>
void compress_decompress(Type *data, Type *ddata, int N, size_t* dims, float eb, float s, size_t& compressed_size){
    if(N==2){
         compress_decompress_2d(data,ddata, dims[0], dims[1], eb, s, compressed_size);
    }
    else if(N==3){
         compress_decompress_3d(data,ddata, dims[0], dims[1], dims[2], eb, s, compressed_size);
    }
    else{
        printf("N=%d is not supported\n", N);
    }
}


extern "C" void compress_decompress_float(float * data, float* ddata,  int N, size_t* dims, float eb, float s, size_t& compressed_size){
    compress_decompress(data, ddata, N, dims, eb, s, compressed_size);
}


// copy from cpSZ end

int main(int argc, char ** argv){
    size_t num_elements = 0;
    float * data = readfile<float>(argv[1], num_elements);
    float eb = atof(argv[2]);
    float s = atof(argv[3]);
    int num_dims = atoi(argv[4]);
    vector<size_t> dimensions(num_dims);
    for(int i=0; i<num_dims; i++){
    	dimensions[num_dims-1-i] = (size_t)atoi(argv[5 + i]);
	//printf(" dim[%d] = %d", num_dims-1-i,dimensions[num_dims-1-i]);
    }
    float max=data[0], min=data[0];
    for (size_t i=0;i<num_elements;i++){
	    if (data[i]>max) max=data[i];
	    if (data[i]<min) min=data[i];
    }
    eb=eb*(max-min);
    using T = float;
    if (s==9) {
    s = std::numeric_limits<T>::infinity();}
    Timer timer;

    if(num_dims == 2){
	    const array<size_t, 2> dims = {dimensions[0], dimensions[1]};
        std::cout<< "dims = " << dims[0] << " " << dims[1] << std::endl;
	    const mgard::TensorMeshHierarchy<2, T> hierarchy(dims);
	    const size_t ndof = hierarchy.ndof();
	    timer.start();
	    mgard::CompressedDataset<2, T> compressed = mgard::compress(hierarchy, data, s, eb);
	    timer.end();
	    printf("mgard compress time = %f \n", timer.get());
	    timer.start();
	    mgard::DecompressedDataset<2, T> decompressed = mgard::decompress(compressed);
	    timer.end();
	    printf("mgard decompress time = %f \n", timer.get());
	    writefile((string(argv[1]) + ".mgard.out").c_str(), decompressed.data(), num_elements);
	    printf("compressionRatio = %.4f\n", 1.0*num_elements*sizeof(T) / compressed.size());
                printf("bitrate = %.4f\n", 32*1.0/(1.0*num_elements*sizeof(T) / compressed.size()));

	    verify(data, decompressed.data(), num_elements);
        // size_t compressed_size = 0;
        // const T* decompressed_data =  compress_decompress_float(data, num_dims, dimensions.data(), atof(argv[2]), s,compressed_size);
        // printf("compressed_size = %d\n", compressed_size);
        // verify(data, decompressed_data, num_elements);

        // std::vector<float> ddata;
        // ddata.resize(num_elements);
        size_t compressed_size = 0;
        // // compress_decompress(data, ddata.data(),num_dims,dimensions.data() , eb, s,compressed_size);

        mgard::DecompressedDataset<2, T> decompresseddata = compress_decompress_2d(data,  dimensions[0],  dimensions[1], eb,  s,  compressed_size);
        writefile((string(argv[1]) + ".mgard.out").c_str(), decompresseddata.data(), num_elements);
        printf("compressionRatio = %.4f\n", 1.0*num_elements*sizeof(T) /compressed_size);
        printf("bitrate = %.4f\n", 32*1.0/(1.0*num_elements*sizeof(T) / compressed_size));

        // verify(data, decompressed.data(), num_elements);


	}
	else if(num_dims == 3){
	    const array<size_t, 3> dims = {dimensions[0], dimensions[1], dimensions[2]};
	    const mgard::TensorMeshHierarchy<3, T> hierarchy(dims);
	    const size_t ndof = hierarchy.ndof();
	    timer.start();
	    mgard::CompressedDataset<3, T> compressed = mgard::compress(hierarchy, data, s, eb);
	    timer.end();
	    printf("mgard compress time = %f \n", timer.get());
	    timer.start();
	    mgard::DecompressedDataset<3, T> decompressed = mgard::decompress(compressed);
	    timer.end();
	    printf("mgard decompress time = %f \n", timer.get());
	    writefile((string(argv[1]) + ".mgard.out").c_str(), decompressed.data(), num_elements);
	    writefile((string(argv[1]) + ".mgard.out").c_str(), decompressed.data(), num_elements);
	    printf("compressionRatio = %.4f\n", 1.0*num_elements*sizeof(T) / compressed.size());
        printf("bitrate = %.4f\n", 32*1.0/(1.0*num_elements*sizeof(T) / compressed.size()));
	    verify(data, decompressed.data(), num_elements);		
	}

}

