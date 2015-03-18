
function rw(n::Int64)
    s::Float32 = 0
    for i = 1:n
        s += rand()
    end
    s
end

function fib(n::Int64)
  if (n <= 2) 
     1
  else
     fib(n-1) + fib(n-2)
  end
end
