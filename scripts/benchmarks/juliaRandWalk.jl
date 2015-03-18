
function rw(n::Int32)
    s::Float32 = 0
    for i = 1:n
        s += rand()
    end
    s
end
