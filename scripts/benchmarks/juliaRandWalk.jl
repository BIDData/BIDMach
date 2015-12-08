
function rw(a)
    n = length(a)
    a[1] = rand() - 0.5
    for i = 2:n
        a[i] = a[i-1] + rand() - 0.5
    end
    a
end

function fib(n::Int64)
  if (n <= 2) 
     1
  else
     fib(n-1) + fib(n-2)
  end
end
