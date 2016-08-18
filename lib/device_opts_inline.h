__forceinline__ static __device__ double2 fetch_double2(cudaTextureObject_t t, int i)
{
  int4 v =tex1Dfetch<int4>(t,i);
  return make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
}

__forceinline__ static __device__ float2 fetch_float2(cudaTextureObject_t t, int i)
{
  float2 v = tex1Dfetch<float2>(t,i);
  return v;
}


template<typename Float2>
__device__ inline Float2 operator*(const Float2 a, const Float2 b){
  Float2 res;
  res.x = a.x*b.x - a.y*b.y;
  res.y = a.x*b.y + a.y*b.x;
  return res;
}


__device__ inline float2 operator*(const float a , const float2 b){
  float2 res;
  res.x = a*b.x;
  res.y = a*b.y;
  return res;
} 


__device__ inline double2 operator*(const double a , const double2 b){
  double2 res;
  res.x = a*b.x;
  res.y = a*b.y;
  return res;
} 


template<typename Float2>
__device__ inline Float2 operator*(const int a , const Float2 b){
  Float2 res;
  res.x = a*b.x;
  res.y = a*b.y;
  return res;
} 

template<typename Float2>
__device__ inline Float2 operator+(const Float2 a, const Float2 b){
  Float2 res;
  res.x = a.x + b.x;
  res.y = a.y + b.y;
  return res;
}

template<typename Float2>
__device__ inline Float2 operator-(const Float2 a, const Float2 b){
  Float2 res;
  res.x = a.x - b.x;
  res.y = a.y - b.y;
  return res;
}

template<typename Float2>
__device__ inline Float2 conj(const Float2 a){
  Float2 res;
  res.x = a.x;
  res.y = -a.y;
  return res;
}

__device__ inline float norm(const float2 a){
  float res;
  res = sqrt(a.x*a.x + a.y*a.y);
  return res;
}

__device__ inline double norm(const double2 a){
  double res;
  res = sqrt(a.x*a.x + a.y*a.y);
  return res;
}

