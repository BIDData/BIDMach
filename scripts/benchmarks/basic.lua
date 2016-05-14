
nreps = 10
n = 10000
a = {}
b = {}
c = {}

for i = 1, n do
  a[i] = {};
  b[i] = {};
  c[i] = {};
  for j = 1, n do	
    a[i][j] = math.random();
    b[i][j] = math.random();
    c[i][j] = 0;
  end
end

t1=os.time();

for irep = 1, nreps do
for i = 1, n do
  for j = 1, n do	
    c[i][j] = a[i][j] + b[i][j];
  end
end
end

t2=os.time();
dt = t2 - t1;
n2 = 1.0*n*n*nreps

print(string.format("time=3.2%f, Mflops=3.2%f",dt/nreps,n2/dt/1e6))
  