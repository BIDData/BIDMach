
n = 10000;
a = rand(Float32,(n,n))
b = rand(Float32,(n,n))
c = zeros(Float32,(n,n));
t1 = time();

for i = 1:n
  for j = 1:n
    c[i,j] = a[i,j] + b[i,j]
  end
end

t2 = time();

for i = 1:10
c = a+ b;	
end

t3 = time();

dt1 = t2 - t1
dt2 = t3 - t2
n2 = n*n

mflops1 = n2 / dt1 / 1e6;
mflops2 = n2 / dt2 / 1e5;

println("times $dt1,$dt2, mflops $mflops1,$mflops2")


