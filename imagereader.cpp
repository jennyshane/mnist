#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>

typedef unsigned char uchar;

int main(int argc, char** argv){

	std::ifstream datafile;
	datafile.open("t10k-images-idx3-ubyte", std::ios::binary);
	char memblock[4];

	std::vector<int> msg;
	for(int n=0; n<4; ++n){
		datafile.read(memblock, 4);
		char revmem[4];
		for(int i=0; i<4; ++i){
			revmem[i]=memblock[3-i];
		}

		std::cout<<*((int*) revmem)<<std::endl;
		msg.push_back(*((int*) revmem));
	}

	if(msg[0]!=2051){
		std::cout<<"Something has gone wrong"<<std::endl;
		datafile.close();
	}
	int imgsz=msg[2]*msg[3];
	int numimgs=msg[1];
	std::vector<std::vector<uchar> > images(numimgs, std::vector<uchar>(imgsz));
	char* imgmem= (char*) malloc(imgsz);

	for(int i=0; i<numimgs; i++){
		datafile.read(imgmem, imgsz);
		std::copy(imgmem, imgmem+imgsz, images[i].begin());
	}
	std::cout<<images.size()<<std::endl;
	free(imgmem);
	datafile.close();	
	for(int n=1; n<argc; n++){
		int idx=atoi(argv[n]);
		if(idx<numimgs){

			std::cout<<"Index "<<idx<<std::endl;
			for(int i=0; i<msg[2]; ++i){
				for(int j=0; j<msg[3]; ++j){
					std::cout<<ceil((float)images[idx][i*msg[3]+j]/255.0);
				}
				std::cout<<std::endl;
			}
		}
	}


	return 0;
}
