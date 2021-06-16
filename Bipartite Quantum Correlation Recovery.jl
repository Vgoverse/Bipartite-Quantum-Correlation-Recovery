
using LinearAlgebra, Plots, QuantumInformation
function Pcalc(E, F, psi)
    P = zeros(Float64,size(E,3),size(F,3),size(E,4),size(F,4))
    psipsi = psi*psi'
    for a=1:size(E,3),  b=1:size(F,3), s=1:size(E,4), t=1:size(F,4)
        P[a,b,s,t] = real(tr(kron(E[:,:,a,s], F[:,:,b,t]) * psipsi))
    end
    return P
end

function CHSH()
    psi = [1/sqrt(2), 0, 0, 1/sqrt(2)]

    E = zeros(ComplexF64,(2,2,2,2))
    E[:,:,1,1] = [1/2+0im 1/2; 1/2 1/2] #E_0^0
    E[:,:,2,1] = [1/2 -1/2; -1/2+0im 1/2]#E_0^1
    E[:,:,1,2] = [1/2 -im/2; im/2 1/2] #E_1^0
    E[:,:,2,2] = [1/2 im/2; -im/2 1/2] #E_1^1

    # squares L2 norm
    F = zeros(ComplexF64,(2,2,2,2))
    F[:,:,1,1] = [1/2 1/(sqrt(2)*2)*(1+im); 1/(sqrt(2)*2)*(1-im) 1/2] #F_0^0
    F[:,:,2,1] = [1/2 -1/(sqrt(2)*2)*(1+im); -1/(sqrt(2)*2)*(1-im) 1/2]  #F_0^1
    F[:,:,1,2] = [1/2 1/(2*sqrt(2))*(1-im); 1/(2*sqrt(2))*(1+im) 1/2] #F_1^0
    F[:,:,2,2] = [1/2 1/(2*sqrt(2))*(-1+im); 1/(2*sqrt(2))*(-1-im) 1/2] #F_1^1

    return E, F, psi
end


function costobj(P, E, F, psi)
    obj = 0.0
    P_calc = Pcalc(E, F, psi)
    for a=1:size(E,3), b=1:size(F,3), s=1:size(E,4), t=1:size(F,4)
        obj +=(P[a,b,s,t] - P_calc[a,b,s,t])^2
    end
    return obj

end

function cost(P, E, F, psi)
    obj = 0.0
    P_calc = Pcalc(E, F, psi)
    for a=1:size(E,3), b=1:size(F,3), s=1:size(E,4), t=1:size(F,4)
        obj +=(P[a,b,s,t] - P_calc[a,b,s,t])^2
    end

    #pentalties
    p_unitpsi = (psi'*psi-1)^2

    p_sumE= 0.0::Float64
    for s=1:size(E,4)
        p_sumE += tr((sum(E[:,:,a1,s] for a1 = 1:size(E,3))-I)*(sum(E[:,:,a2,s] for a2 = 1:size(E,3))-I))
    end
    p_sumF = 0.0::Float64
    for t=1:size(F,4)
        p_sumF += tr((sum(F[:,:,b1,t] for b1 = 1:size(F,3))-I)*(sum(F[:,:,b2,t] for b2 = 1:size(F,3))-I))
    end

    penalties = real(p_unitpsi +p_sumE +p_sumF)
    return(real(obj + penalties), p_unitpsi, p_sumE, p_sumF,obj,penalties)
end
function costPVM(P, E, F, psi)

    obj = 0.0
    P_calc = Pcalc(E, F, psi)
    for a=1:size(E,3), b=1:size(F,3), s=1:size(E,4), t=1:size(F,4)
        obj +=(P[a,b,s,t] - P_calc[a,b,s,t])^2
    end

    #pentalties
    p_unitpsi = (psi'*psi-1)^2

    p_sumE= 0.0::Float64
    for s=1:size(E,4)
        p_sumE += tr((sum(E[:,:,a1,s] for a1 = 1:size(E,3))-I)*(sum(E[:,:,a2,s] for a2 = 1:size(E,3))-I))
    end
    p_PVME = 0.0
    for s=1:size(E,4),a=1:size(E,3)
        p_PVME += tr((E[:,:,a,s]^2-E[:,:,a,s])*((E[:,:,a,s]^2-E[:,:,a,s])))
    end

    p_sumF = 0.0::Float64
    for t=1:size(F,4)
        p_sumF += tr((sum(F[:,:,b1,t] for b1 = 1:size(F,3))-I)*(sum(F[:,:,b2,t] for b2 = 1:size(F,3))-I))
    end
    p_PVMF = 0.0
    for t=1:size(F,4),b=1:size(F,3)
        p_PVMF += tr((F[:,:,b,t]^2-F[:,:,b,t])*((F[:,:,b,t]^2-F[:,:,b,t])))
    end

    penalties = real(p_unitpsi +p_sumE +p_sumF+ p_PVME+ p_PVMF)
    return(real(obj + penalties), p_unitpsi, p_sumE, p_sumF,obj,penalties)
end
function single!(i,j,d,E)
    E .= zeros(ComplexF64,d,d)
    E[i,j]=1
    E
end

function delta(i,j)
    if i == j
        return 1.0
    else
        return 0.0
    end
end
function gradientNOPEN!(P, X, Y, psi, dX, dY, dpsi, d, Pa, Pb, Ps, Pt)
    E = zeros(ComplexF64,d,d,Pa,Ps) #matrix of matrici
    F = zeros(ComplexF64,d,d,Pb,Pt)
    # add full cost function
    for i1 = 1:Pa, i2 = 1:Ps
        E[:,:,i1,i2] = adjoint(X[:,:,i1,i2])*X[:,:,i1,i2]
    end
    for i1 = 1:Pb, i2 = 1:Pt
        F[:,:,i1,i2] = adjoint(Y[:,:,i1,i2])*Y[:,:,i1,i2]
    end

    #dX :

    P_calc = Pcalc(E, F, psi)
    Single = zeros(ComplexF64,d,d)

    for a = 1:Pa, s = 1:Ps #X_a^s
        for k = 1:d, l in 1:k #row,column of a,s matrix
            dX[k,l,a,s] = -4*sum(X[k,i,a,s]*(P[a,b,s,t]-P_calc[a,b,s,t])* dot(psi[(((l-1)*d)+1):(l*d)],F[:,:,b,t],psi[((i-1)*d+1):(i*d)]) for i=1:d,b=1:Pb,t=1:Pt)
        end
    end

    for b = 1:Pb, t = 1:Pt #Y_b^t
        for k = 1:d, l in 1:k #row,column of a,s matrix
            dY[k,l,b,t] = -4*sum(Y[k,i,b,t]*(P[a,b,s,t]-P_calc[a,b,s,t])* dot(psi[l:d:(d^2 - d)+ l], E[:,:,a,s], psi[i:d:((d^2-d)+i )]) for i=1:d,a=1:Pa,s=1:Ps)
        end
    end
    dpsi .= zeros(ComplexF64,d^2)
    for a =1:Pa, s =1:Ps, b= 1:Pb, t= 1:Pt
        kronEF= (kron(E[:,:,a,s],F[:,:,b,t])*psi)
        dpsi .+=  -4 * (P[a,b,s,t] - P_calc[a,b,s,t]) * kronEF
    end

end
function gradient!(P, X, Y, psi, dX, dY, dpsi, d, Pa, Pb, Ps, Pt)
    E = zeros(ComplexF64,d,d,Pa,Ps) #matrix of matrici
    F = zeros(ComplexF64,d,d,Pb,Pt)
    # add full cost function
    for i1 = 1:Pa, i2 = 1:Ps
        E[:,:,i1,i2] = adjoint(X[:,:,i1,i2])*X[:,:,i1,i2]
    end
    for i1 = 1:Pb, i2 = 1:Pt
        F[:,:,i1,i2] = adjoint(Y[:,:,i1,i2])*Y[:,:,i1,i2]
    end

    #dX :

    P_calc = Pcalc(E, F, psi)
    Single = zeros(ComplexF64,d,d)

    for a = 1:Pa, s = 1:Ps #X_a^s
        for k = 1:d, l = 1:k #row,column of a,s matrix
            dX[k,l,a,s] = -4*sum(X[k,i,a,s]*(P[a,b,s,t]-P_calc[a,b,s,t])* dot(psi[(((l-1)*d)+1):(l*d)],F[:,:,b,t],psi[((i-1)*d+1):(i*d)]) for i=1:d,b=1:Pb,t=1:Pt)
            dX[k,l,a,s]+=4*sum(X[k,i,a,s] * (sum(E[i,l,a2,s] for a2 = 1:Pa)-delta(i,l)) for i = 1:d)
        end
    end

    for b = 1:Pb, t = 1:Pt #Y_b^t
        for k = 1:d, l = 1:k #row,column of a,s matrix
            dY[k,l,b,t] = -4*sum(Y[k,i,b,t]*(P[a,b,s,t]-P_calc[a,b,s,t])* dot(psi[l:d:(d^2 - d)+ l], E[:,:,a,s], psi[i:d:((d^2-d)+i )]) for i=1:d,a=1:Pa,s=1:Ps)
            dY[k,l,b,t]+=4*sum(Y[k,i,b,t] * (sum(F[i,l,b2,t] for b2 = 1:Pb)-delta(i,l)) for i = 1:d)
        end
    end


    dpsi .= (4* ((psi'*psi) -1)*psi)
    for a =1:Pa, s =1:Ps, b= 1:Pb, t= 1:Pt
        kronEF= (kron(E[:,:,a,s],F[:,:,b,t])*psi)
        dpsi .+=  -4 * (P[a,b,s,t] - P_calc[a,b,s,t]) * kronEF
    end

end

function gradientPVM!(P, X, Y, psi, dX, dY, dpsi, d, Pa, Pb, Ps, Pt)
    E = zeros(ComplexF64,d,d,Pa,Ps) #matrix of matrici
    F = zeros(ComplexF64,d,d,Pb,Pt)
    # add full cost function
    for i1 = 1:Pa, i2 = 1:Ps
        E[:,:,i1,i2] = adjoint(X[:,:,i1,i2])*X[:,:,i1,i2]
    end
    for i1 = 1:Pb, i2 = 1:Pt
        F[:,:,i1,i2] = adjoint(Y[:,:,i1,i2])*Y[:,:,i1,i2]
    end

    #dX :

    P_calc = Pcalc(E, F, psi)
    Single = zeros(ComplexF64,d,d)

    for a = 1:Pa, s = 1:Ps #X_a^s
        for k = 1:d, l = 1:k #row,column of a,s matrix
            dX[k,l,a,s] = -4*sum(X[k,i,a,s]*(P[a,b,s,t]-P_calc[a,b,s,t])* dot(psi[(((l-1)*d)+1):(l*d)],F[:,:,b,t],psi[((i-1)*d+1):(i*d)]) for i=1:d,b=1:Pb,t=1:Pt)
            dX[k,l,a,s]+=4*sum(X[k,i,a,s] * (sum(E[i,l,a2,s] for a2 = 1:Pa)-delta(i,l)) for i = 1:d)
            dX[k,l,a,s]+=2*(4*tr(single!(l,k,d,Single)*X[:,:,a,s]*E[:,:,a,s]^3) + 2*tr(single!(l,k,d,Single)*X[:,:,a,s]*E[:,:,a,s]) -
            6*tr(single!(l,k,d,Single)*X[:,:,a,s]*E[:,:,a,s]^2)   )
        end
    end

    for b = 1:Pb, t = 1:Pt #Y_b^t
        for k = 1:d, l = 1:k #row,column of a,s matrix
            dY[k,l,b,t] = -4*sum(Y[k,i,b,t]*(P[a,b,s,t]-P_calc[a,b,s,t])* dot(psi[l:d:(d^2 - d)+ l], E[:,:,a,s], psi[i:d:((d^2-d)+i )]) for i=1:d,a=1:Pa,s=1:Ps)
            dY[k,l,b,t]+=4*sum(Y[k,i,b,t] * (sum(F[i,l,b2,t] for b2 = 1:Pb)-delta(i,l)) for i = 1:d)
            dY[k,l,b,t]+=2*(4*tr(single!(l,k,d,Single)*Y[:,:,b,t]*F[:,:,b,t]^3) + 2*tr(single!(l,k,d,Single)*Y[:,:,b,t]*F[:,:,b,t]) -
            6*tr(single!(l,k,d,Single)*Y[:,:,b,t]*F[:,:,b,t]^2)   )
        end
    end


    dpsi .= (4* ((psi'*psi) -1)*psi)
    for a =1:Pa, s =1:Ps, b= 1:Pb, t= 1:Pt
        kronEF= (kron(E[:,:,a,s],F[:,:,b,t])*psi)
        dpsi .+=  -4 * (P[a,b,s,t] - P_calc[a,b,s,t]) * kronEF
    end

end

function gradientNOPSI!(P, X, Y, psi, dX, dY, dpsi, d, Pa, Pb, Ps, Pt)
    E = zeros(ComplexF64,d,d,Pa,Ps) #matrix of matrici
    F = zeros(ComplexF64,d,d,Pb,Pt)
    # add full cost function
    for i1 = 1:Pa, i2 = 1:Ps
        E[:,:,i1,i2] = adjoint(X[:,:,i1,i2])*X[:,:,i1,i2]
    end
    for i1 = 1:Pb, i2 = 1:Pt
        F[:,:,i1,i2] = adjoint(Y[:,:,i1,i2])*Y[:,:,i1,i2]
    end

    P_calc = Pcalc(E, F, psi)
    Single = zeros(ComplexF64,d,d)

    for a = 1:Pa, s = 1:Ps #X_a^s
        for k = 1:d, l = 1:k #row,column of a,s matrix
            dX[k,l,a,s] = -4*sum(X[k,i,a,s]*(P[a,b,s,t]-P_calc[a,b,s,t])* dot(psi[(((l-1)*d)+1):(l*d)],F[:,:,b,t],psi[((i-1)*d+1):(i*d)]) for i=1:d,b=1:Pb,t=1:Pt)
            dX[k,l,a,s]+=4*sum(X[k,i,a,s] * (sum(E[i,l,a2,s] for a2 = 1:Pa)-delta(i,l)) for i = 1:d)
        end
    end

    for b = 1:Pb, t = 1:Pt #Y_b^t
        for k = 1:d, l = 1:k #row,column of a,s matrix
            dY[k,l,b,t] = -4*sum(Y[k,i,b,t]*(P[a,b,s,t]-P_calc[a,b,s,t])* dot(psi[l:d:(d^2 - d)+ l], E[:,:,a,s], psi[i:d:((d^2-d)+i )]) for i=1:d,a=1:Pa,s=1:Ps)
            dY[k,l,b,t]+=4*sum(Y[k,i,b,t] * (sum(F[i,l,b2,t] for b2 = 1:Pb)-delta(i,l)) for i = 1:d)
        end
    end
end

function gradientsquare(dX,dY,dpsi)
    som = 0.0
    for a = 1:size(dX,3), s = 1:size(dX,4), i = 1:size(dX,1), j = 1:size(dX,2)
        som += abs2(dX[i,j,a,s])
    end
    for b = 1:size(dY,3), t = 1:size(dY,4),  i = 1:size(dY,1), j = 1:size(dY,2)
        som += abs2(dY[i,j,b,t])
    end
    som += real(dpsi'*dpsi)
    som
end



function gradientdescent(P, d; alpha = 0.5, beta=0.9)
    println("----------------------------------------------------------------------------------")
    println("\n\n\n\n\n")
    println("New Gradient descent")

    Pa= size(P,1)
    Pb= size(P,2)
    Ps= size(P,3)
    Pt= size(P,4)

    println("a $Pa,b $Pb, s $Ps, t $Pt, d $d")
    # random initialization
    psi = randn(ComplexF64,d^2)
    psi /= norm(psi)

    X = zeros(ComplexF64,d,d,Pa,Ps)  #dimensies uit p halen
    Y = zeros(ComplexF64,d,d,Pb,Pt)
    for a = 1:Pa,s =1:Ps,i=1:d, j=1:i
        X[i,j,a,s] = randn(ComplexF64)/d
    end
    for b = 1:Pb,t =1:Pt,i=1:d, j=1:i
        Y[i,j,b,t] = randn(ComplexF64)/d
    end

    dX = zeros(ComplexF64,d,d,Pa,Ps)
    dY = zeros(ComplexF64,d,d,Pb,Pt)
    dpsi = zeros(ComplexF64, d^2)

    Xtest = zeros(ComplexF64,d,d,Pa,Ps)
    Ytest = zeros(ComplexF64,d,d,Pb,Pt)
    psitest = zeros(ComplexF64, d^2)

    E = zeros(ComplexF64,d,d,Pa,Ps) #matrix of matrici
    F = zeros(ComplexF64,d,d,Pb,Pt)

    Etest = zeros(ComplexF64,d,d,Pa,Ps) #matrix of matrici
    Ftest = zeros(ComplexF64,d,d,Pb,Pt)
    c_array = []
    t1 = 1
    t3 = 0 #counter
    while t1< 1000002
        gradient!(P, X, Y, psi, dX, dY, dpsi, d, Pa, Pb, Ps, Pt)

        for a = 1:Pa, s = 1:Ps
            E[:,:,a,s] = adjoint(X[:,:,a,s])*X[:,:,a,s]
        end
        for b = 1:Pb, t = 1:Pt
            F[:,:,b,t] = adjoint(Y[:,:,b,t])*Y[:,:,b,t]
        end
        c = cost(P, E, F, psi)
        if t1 % 100 ==1
            println(t1,c) #real(obj + penalties), p_unitpsi, p_sumE, p_sumF)
        end
        if t1 % 100 == 0
            display(plot(1:size(c_array,1), c_array,xaxis=:log, yaxis=:log, legend = false, title = "Gradient descent backward linesearch, P = C_6", xlabel = "Interations (n)", ylabel = "Cost function f(x_k)"))
        end
        t2 =1.0::Float64
        while true
            Xtest = X - t2.*dX
            Ytest = Y - t2.*dY
            psitest = psi - t2.*dpsi

            for a = 1:Pa, s = 1:Ps
                Etest[:,:,a,s] = (Xtest[:,:,a,s]')*Xtest[:,:,a,s]
            end

            for b = 1:Pb, t = 1:Pt
                Ftest[:,:,b,t] = (Ytest[:,:,b,t]')*Ytest[:,:,b,t]
            end

            if (cost(P,Etest,Ftest,psitest)[1]) < c[1]- alpha*t2 *gradientsquare(dX,dY,dpsi)
                X = Xtest
                Y = Ytest
                psi = psitest
                push!(c_array,c[1])
                break
            else
                t2 *= beta
                if t2 < 10.0^(-10)
                    X -= dX
                    Y -= dY
                    psi -=  dpsi
                    print("t2 heel klein")
                    return(c[1],t1)
                    break
                end
            end

        end
        if abs(c[1]) < 1e-5
            display(plot(1:size(c_array,1), c_array,xaxis=:log, yaxis=:log, legend = false, title = "Gradient descent backward linesearch, P = C_6", xlabel = "Interations (n)", ylabel = "Cost function f(x_k)"))
            println("E, $E")
            println("X , $X")
            println("F, $F")
            println("Y, $Y")
            println("psi $psi")
            output_file = open("output_file.jl","w") # this will create a file named output_file.jl, where we will write the data
            write(output_file, "\n\n\n\n a:$Pa b:$Pb s:$Ps t:$Pt d:$d; \n \n")
            write(output_file, "X = ") # writes A =
            show(output_file, X) # writes the content of A
            write(output_file, "; \n \n") # puts a semicolon to suppress the output and two line breaks
            write(output_file, "E = ") # writes A =
            show(output_file, E) # writes the content of A
            write(output_file, "; \n \n") # puts a semicolon to suppress the output and two line breaks
            write(output_file, "Y = ") # writes A =
            show(output_file, Y) # writes the content of A
            write(output_file, "; \n \n") # puts a semicolon to suppress the output and two line breaks
            write(output_file, "F = ") # writes A =
            show(output_file, F) # writes the content of A
            write(output_file, "; \n \n") # puts a semicolon to suppress the output and two line breaks
            write(output_file, "psi = ") # writes A =
            show(output_file, psi) # writes the content of A
            write(output_file, "; \n \n") # puts a semicolon to suppress the output and two line breaks
            close(output_file)
            println(t1)
            return(c[1])
        end
        if isnan(c[1])
            println("NaN, annuleer die hele zooi maar")
            return(10)
        end
        t1+=1
    end

    return(c[1])
    println("E, $E")
    println("X , $X")
    println("F, $F")
    println("Y, $Y")
    println("psi $psi")
end

function gradientdescentPVM(P, d; alpha = 0.5, beta=0.9, Pname = "P")
    println("----------------------------------------------------------------------------------")
    println("\n\n\n\n\n")
    println("New Gradient descent")

    Pa= size(P,1)
    Pb= size(P,2)
    Ps= size(P,3)
    Pt= size(P,4)

    println("a $Pa,b $Pb, s $Ps, t $Pt, d $d")
    # random initialization
    psi = randn(ComplexF64,d^2)
    psi /= norm(psi)

    X = zeros(ComplexF64,d,d,Pa,Ps)  #dimensies uit p halen
    Y = zeros(ComplexF64,d,d,Pb,Pt)
    for a = 1:Pa,s =1:Ps,i=1:d, j=1:i
        X[i,j,a,s] = randn(ComplexF64)/d
    end
    for b = 1:Pb,t =1:Pt,i=1:d, j=1:i
        Y[i,j,b,t] = randn(ComplexF64)/d
    end


    dX = zeros(ComplexF64,d,d,Pa,Ps)
    dY = zeros(ComplexF64,d,d,Pb,Pt)
    dpsi = zeros(ComplexF64, d^2)

    Xtest = zeros(ComplexF64,d,d,Pa,Ps)
    Ytest = zeros(ComplexF64,d,d,Pb,Pt)
    psitest = zeros(ComplexF64, d^2)

    E = zeros(ComplexF64,d,d,Pa,Ps) #matrix of matrici
    F = zeros(ComplexF64,d,d,Pb,Pt)

    Etest = zeros(ComplexF64,d,d,Pa,Ps) #matrix of matrici
    Ftest = zeros(ComplexF64,d,d,Pb,Pt)
    c_array = []
    t1 = 1
    t3 = 0 #counter
    while t1< 1000002
        gradientPVM!(P, X, Y, psi, dX, dY, dpsi, d, Pa, Pb, Ps, Pt)

        for a = 1:Pa, s = 1:Ps
            E[:,:,a,s] = adjoint(X[:,:,a,s])*X[:,:,a,s]
        end
        for b = 1:Pb, t = 1:Pt
            F[:,:,b,t] = adjoint(Y[:,:,b,t])*Y[:,:,b,t]
        end
        c = costPVM(P, E, F, psi)
        if t1 % 100 ==1
            println(t1,c) #real(obj + penalties), p_unitpsi, p_sumE, p_sumF)
        end
        if t1 % 100 == 0
            display(plot(1:size(c_array,1), c_array,xaxis=:log, yaxis=:log, title = "gradient descent $Pname", xlabel = "interations (n)", ylabel = "cost function"))
        end
        t2 =1.0::Float64
        while true
            Xtest = X - t2.*dX
            Ytest = Y - t2.*dY
            psitest = psi - t2.*dpsi

            for a = 1:Pa, s = 1:Ps
                Etest[:,:,a,s] = (Xtest[:,:,a,s]')*Xtest[:,:,a,s]
            end

            for b = 1:Pb, t = 1:Pt
                Ftest[:,:,b,t] = (Ytest[:,:,b,t]')*Ytest[:,:,b,t]
            end

            if (costPVM(P,Etest,Ftest,psitest)[1]) < c[1]- alpha*t2 *gradientsquare(dX,dY,dpsi)
                X = Xtest
                Y = Ytest
                psi = psitest
                push!(c_array,c[1])
                break
            else
                t2 *= beta
                if t2 < 10.0^(-20)
                    X -= dX
                    Y -= dY
                    psi -=  dpsi
                    print("t2 heel klein")
                    return(c[1],t1)
                    break
                end
            end

        end

        if abs(c[1]) < 1e-5
            display(plot(1:size(c_array,1), c_array,xaxis=:log, yaxis=:log, title = "gradient descent PCHSH", xlabel = "interations (n)", ylabel = "cost function"))
            println("Geweldig gedaan!! amount of iterations: $t1")
            println("E, $E")
            println("X , $X")
            println("F, $F")
            println("Y, $Y")
            println("psi $psi")
            output_file = open("output_file.jl","w") # this will create a file named output_file.jl, where we will write the data
            write(output_file, "\n\n\n\n a:$Pa b:$Pb s:$Ps t:$Pt d:$d; \n \n")
            write(output_file, "X = ") # writes A =
            show(output_file, X) # writes the content of A
            write(output_file, "; \n \n") # puts a semicolon to suppress the output and two line breaks
            write(output_file, "E = ") # writes A =
            show(output_file, E) # writes the content of A
            write(output_file, "; \n \n") # puts a semicolon to suppress the output and two line breaks
            write(output_file, "Y = ") # writes A =
            show(output_file, Y) # writes the content of A
            write(output_file, "; \n \n") # puts a semicolon to suppress the output and two line breaks
            write(output_file, "F = ") # writes A =
            show(output_file, F) # writes the content of A
            write(output_file, "; \n \n") # puts a semicolon to suppress the output and two line breaks
            write(output_file, "psi = ") # writes A =
            show(output_file, psi) # writes the content of A
            write(output_file, "; \n \n") # puts a semicolon to suppress the output and two line breaks
            close(output_file)
            return(c[1])
        end
        if isnan(c[1])
            println("NaN, annuleer die hele zooi maar")
            return(10)
        end

        t1+=1
    end

    return(c[1])
    println("E, $E")
    println("X , $X")
    println("F, $F")
    println("Y, $Y")
    println("psi $psi")
end

function gradientdescentNOPSI(P, d; alpha = 0.5, beta=0.9)
    println("----------------------------------------------------------------------------------")
    println("\n\n\n\n\n")
    println("New Gradient descent")



    Pa= size(P,1)
    Pb= size(P,2)
    Ps= size(P,3)
    Pt= size(P,4)

    println("a $Pa,b $Pb, s $Ps, t $Pt, d $d")
    # random initialization
    psi = zeros(ComplexF64,d^2)
    psi[1] = 1/sqrt(2)
    psi[d^2] = 1/sqrt(2)

    X = zeros(ComplexF64,d,d,Pa,Ps)  #dimensies uit p halen
    Y = zeros(ComplexF64,d,d,Pb,Pt)
    for a = 1:Pa,s =1:Ps,i=1:d, j=1:i
        X[i,j,a,s] = randn(ComplexF64)/d
    end
    for b = 1:Pb,t =1:Pt,i=1:d, j=1:i
        Y[i,j,b,t] = randn(ComplexF64)/d
    end


    dX = zeros(ComplexF64,d,d,Pa,Ps)
    dY = zeros(ComplexF64,d,d,Pb,Pt)
    dpsi = zeros(ComplexF64, d^2)

    Xtest = zeros(ComplexF64,d,d,Pa,Ps)
    Ytest = zeros(ComplexF64,d,d,Pb,Pt)
    psitest = zeros(ComplexF64, d^2)

    E = zeros(ComplexF64,d,d,Pa,Ps) #matrix of matrici
    F = zeros(ComplexF64,d,d,Pb,Pt)

    Etest = zeros(ComplexF64,d,d,Pa,Ps) #matrix of matrici
    Ftest = zeros(ComplexF64,d,d,Pb,Pt)
    c_array = []
    t1 = 1
    t3 = 0 #counter
    while t1< 1000002
        gradientNOPSI!(P, X, Y, psi, dX, dY, dpsi, d, Pa, Pb, Ps, Pt)

        for a = 1:Pa, s = 1:Ps
            E[:,:,a,s] = adjoint(X[:,:,a,s])*X[:,:,a,s]
        end
        for b = 1:Pb, t = 1:Pt
            F[:,:,b,t] = adjoint(Y[:,:,b,t])*Y[:,:,b,t]
        end
        c = cost(P, E, F, psi)
        if t1 % 100 ==1
            println(t1,c) #real(obj + penalties), p_unitpsi, p_sumE, p_sumF)
        end
        if t1 % 100 == 0
            display(plot(1:size(c_array,1), c_array,xaxis=:log, yaxis=:log))
        end
        t2 =1.0::Float64
        while true
            Xtest = X - t2.*dX
            Ytest = Y - t2.*dY
            # psitest = psi - t2.*dpsi

            for a = 1:Pa, s = 1:Ps
                Etest[:,:,a,s] = (Xtest[:,:,a,s]')*Xtest[:,:,a,s]
            end

            for b = 1:Pb, t = 1:Pt
                Ftest[:,:,b,t] = (Ytest[:,:,b,t]')*Ytest[:,:,b,t]
            end

            if (cost(P,Etest,Ftest,psi)[1]) < c[1]- alpha*t2 *gradientsquare(dX,dY,dpsi)
                X = Xtest
                Y = Ytest
                push!(c_array,c[1])
                break
            else
                t2 *= beta
                if t2 < 10.0^(-10)
                    X -= dX
                    Y -= dY
                    psi -=  dpsi
                    print("t2 heel klein")
                    return(c[1],t1)
                    break

                end
            end

        end

        if abs(c[1]) < 1e-5
            display(plot(1:(t1), c_array,xaxis=:log, yaxis=:log))
            println("Geweldig gedaan!! amount of iterations: $t1")
            println("E, $E")
            println("X , $X")
            println("F, $F")
            println("Y, $Y")
            println("psi $psi")
            output_file = open("output_file.jl","w") # this will create a file named output_file.jl, where we will write the data
            write(output_file, "\n\n\n\n a:$Pa b:$Pb s:$Ps t:$Pt d:$d; \n \n")
            write(output_file, "X = ") # writes A =
            show(output_file, X) # writes the content of A
            write(output_file, "; \n \n") # puts a semicolon to suppress the output and two line breaks
            write(output_file, "E = ") # writes A =
            show(output_file, E) # writes the content of A
            write(output_file, "; \n \n") # puts a semicolon to suppress the output and two line breaks
            write(output_file, "Y = ") # writes A =
            show(output_file, Y) # writes the content of A
            write(output_file, "; \n \n") # puts a semicolon to suppress the output and two line breaks
            write(output_file, "F = ") # writes A =
            show(output_file, F) # writes the content of A
            write(output_file, "; \n \n") # puts a semicolon to suppress the output and two line breaks
            write(output_file, "psi = ") # writes A =
            show(output_file, psi) # writes the content of A
            write(output_file, "; \n \n") # puts a semicolon to suppress the output and two line breaks
            close(output_file)
            return(c[1])
        end
        if isnan(c[1])
            println("NaN, annuleer die hele zooi maar")
            return(10)
        end
        t1+=1
    end

    return(c[1])
    println("E, $E")
    println("X , $X")
    println("F, $F")
    println("Y, $Y")
    println("psi $psi")
end
function gradientdescentNOPEN(P, d; alpha = 0.5, beta=0.95)
    println("----------------------------------------------------------------------------------")
    println("\n\n\n\n\n")
    println("New Gradient descent")



    Pa= size(P,1)
    Pb= size(P,2)
    Ps= size(P,3)
    Pt= size(P,4)

    println("a $Pa,b $Pb, s $Ps, t $Pt, d $d")
    # random initialization
    psi = rand(ComplexF64,d^2)
    psi /= norm(psi)

    X = rand(ComplexF64,d,d,Pa,Ps) ./d #dimensies uit p halen
    Y = rand(ComplexF64,d,d,Pb,Pt) ./d

    dX = zeros(ComplexF64,d,d,Pa,Ps)
    dY = zeros(ComplexF64,d,d,Pb,Pt)
    dpsi = zeros(ComplexF64, d^2)

    Xtest = zeros(ComplexF64,d,d,Pa,Ps)
    Ytest = zeros(ComplexF64,d,d,Pb,Pt)
    psitest = zeros(ComplexF64, d^2)

    E = zeros(ComplexF64,d,d,Pa,Ps) #matrix of matrici
    F = zeros(ComplexF64,d,d,Pb,Pt)

    Etest = zeros(ComplexF64,d,d,Pa,Ps) #matrix of matrici
    Ftest = zeros(ComplexF64,d,d,Pb,Pt)

    t1 = 1
    t3 = 0 #counter
    while t1< 1002
        gradientNOPEN!(P, X, Y, psi, dX, dY, dpsi, d, Pa, Pb, Ps, Pt)

        for a = 1:Pa, s = 1:Ps
            E[:,:,a,s] = adjoint(X[:,:,a,s])*X[:,:,a,s]
        end
        for b = 1:Pb, t = 1:Pt
            F[:,:,b,t] = adjoint(Y[:,:,b,t])*Y[:,:,b,t]
        end
        c = costobj(P, E, F, psi)

        println(t1,c) #real(obj + penalties), p_unitpsi, p_sumE, p_sumF)
        t2 =1.0
        while true
            Xtest = X - t2.*dX
            Ytest = Y - t2.*dY
            psitest = psi - t2.*dpsi

            for a = 1:Pa, s = 1:Ps
                Etest[:,:,a,s] = (Xtest[:,:,a,s]')*Xtest[:,:,a,s]
            end

            for b = 1:Pb, t = 1:Pt
                Ftest[:,:,b,t] = (Ytest[:,:,b,t]')*Ytest[:,:,b,t]
            end

            if (costobj(P,Etest,Ftest,psitest)[1]) < c[1]- alpha*t2 *gradientsquare(dX,dY,dpsi)
                X = Xtest
                Y = Ytest
                psi = psitest
                break
            else
                t2 *= beta
                if t2 < 10.0^(-20)
                    X -= dX
                    Y -= dY
                    psi -=  dpsi
                    print("t2 heel klein")
                    t3 += 1
                    if t3 > 10
                        return(10)
                    end
                    break
                end
            end

        end

        # println("E, $E")
        # println("F, $F")
        if abs(c[1]) < 1e-8

            println("Geweldig gedaan!! amount of iterations: $t1")
            println("E, $E")
            println("X , $X")
            println("F, $F")
            println("Y, $Y")
            println("psi $psi")
            output_file = open("output_file.jl","w") # this will create a file named output_file.jl, where we will write the data
            write(output_file, "\n\n\n\n NOPEN a:$Pa b:$Pb s:$Ps t:$Pt d:$d; \n \n")
            write(output_file, "X = ") # writes A =
            show(output_file, X) # writes the content of A
            write(output_file, "; \n \n") # puts a semicolon to suppress the output and two line breaks
            write(output_file, "E = ") # writes A =
            show(output_file, E) # writes the content of A
            write(output_file, "; \n \n") # puts a semicolon to suppress the output and two line breaks
            write(output_file, "Y = ") # writes A =
            show(output_file, Y) # writes the content of A
            write(output_file, "; \n \n") # puts a semicolon to suppress the output and two line breaks
            write(output_file, "F = ") # writes A =
            show(output_file, F) # writes the content of A
            write(output_file, "; \n \n") # puts a semicolon to suppress the output and two line breaks
            write(output_file, "psi = ") # writes A =
            show(output_file, psi) # writes the content of A
            write(output_file, "; \n \n") # puts a semicolon to suppress the output and two line breaks
            close(output_file)
            return(c[1])
        end
        if isnan(c[1])
            println("NaN, annuleer die hele zooi maar")
            return(10)
        end
        t1+=1
    end
    return(c[1])
    println("E, $E")
    println("X , $X")
    println("F, $F")
    println("Y, $Y")
    println("psi $psi")
end

function gradientdescentMomentum(P, d; t= 1/20, gamma = 0.9)
    println("----------------------------------------------------------------------------------")
    println("\n\n\n\n\n")
    println("New Gradient descent no backward linesearch")


    Pa= size(P,1)
    Pb= size(P,2)
    Ps= size(P,3)
    Pt= size(P,4)

    println("a $Pa,b $Pb, s $Ps, t $Pt, d $d")
    # random initialization
    psi = rand(ComplexF64,d^2)
    psi /= norm(psi)

    X = rand(ComplexF64,d,d,Pa,Ps) ./d #dimensies uit p halen
    Y = rand(ComplexF64,d,d,Pb,Pt) ./d

    dX = zeros(ComplexF64,d,d,Pa,Ps)
    dY = zeros(ComplexF64,d,d,Pb,Pt)
    dpsi = zeros(ComplexF64, d^2)


    E = zeros(ComplexF64,d,d,Pa,Ps) #matrix of matrici
    F = zeros(ComplexF64,d,d,Pb,Pt)

    XM = zeros(ComplexF64,d,d,Pa,Ps)
    YM = zeros(ComplexF64,d,d,Pb,Pt)
    psiM = zeros(ComplexF64, d^2)
    c_array = []

    t1 = 1
    while t1< 1000000
        gradient!(P, X, Y, psi, dX, dY, dpsi, d, Pa, Pb, Ps, Pt)

        for a = 1:Pa, s = 1:Ps
            E[:,:,a,s] = adjoint(X[:,:,a,s])*X[:,:,a,s]
        end
        for b = 1:Pb, t = 1:Pt
            F[:,:,b,t] = adjoint(Y[:,:,b,t])*Y[:,:,b,t]
        end

        c = cost(P, E, F, psi)
        if t1%100==1
            println(t1,c)  #real(obj + penalties), p_unitpsi, p_sumE, p_sumF)
        end
        push!(c_array,c[1])

        XM = gamma * XM + t*dX
        YM = gamma * YM + t*dY
        psiM = gamma * psiM + t*dpsi

        X -= XM
        Y -= YM
        psi -= psiM



        # println("E, $E")
        # println("F, $F")
        if abs(c[1]) < 1e-5
            println("Geweldig gedaan!! amount of iterations: $t1")
            display(plot(1:size(c_array,1), c_array,xaxis=:log, yaxis=:log, legend = false, title = "Gradient descent with fixed stepsize, P = CHSH", xlabel = "Interations (n)", ylabel = "Cost function f(x_k)"))
            return c[1]
        end
        if isnan(c[1])
            println("NaN, annuleer die hele zooi maar")
            return c[1]
        end
        t1+=1
    end

    return c_array
end

function gradientdescentNOLINE(P, d; t = 1/20)
    println("----------------------------------------------------------------------------------")
    println("\n\n\n\n\n")
    println("New Gradient descent no backward linesearch")


    Pa= size(P,1)
    Pb= size(P,2)
    Ps= size(P,3)
    Pt= size(P,4)

    println("a $Pa,b $Pb, s $Ps, t $Pt, d $d")
    # random initialization
    psi = rand(ComplexF64,d^2)
    psi /= norm(psi)

    X = rand(ComplexF64,d,d,Pa,Ps) ./d #dimensies uit p halen
    Y = rand(ComplexF64,d,d,Pb,Pt) ./d

    dX = zeros(ComplexF64,d,d,Pa,Ps)
    dY = zeros(ComplexF64,d,d,Pb,Pt)
    dpsi = zeros(ComplexF64, d^2)


    E = zeros(ComplexF64,d,d,Pa,Ps) #matrix of matrici
    F = zeros(ComplexF64,d,d,Pb,Pt)

    c_array = []

    t1 = 1
    while t1< 10000
        gradient!(P, X, Y, psi, dX, dY, dpsi, d, Pa, Pb, Ps, Pt)

        for a = 1:Pa, s = 1:Ps
            E[:,:,a,s] = adjoint(X[:,:,a,s])*X[:,:,a,s]
        end
        for b = 1:Pb, t = 1:Pt
            F[:,:,b,t] = adjoint(Y[:,:,b,t])*Y[:,:,b,t]
        end

        c = cost(P, E, F, psi)
        if t1%100==1
            @info c #real(obj + penalties), p_unitpsi, p_sumE, p_sumF)
        end
        push!(c_array,c[1])

        X -= dX*t
        Y -= dY*t
        psi -= dpsi*t

        if abs(c[1]) < 1e-5
            println("Geweldig gedaan!! amount of iterations: $t1")
            display(plot(1:size(c_array,1), c_array,xaxis=:log, yaxis=:log, legend = false, title = "Gradient descent with fixed stepsize, P = CHSH", xlabel = "Interations (n)", ylabel = "Cost function f(x_k)"))
            return c_array
        end
        if isnan(c[1])
            println("NaN, annuleer die hele zooi maar")
            return c[1]
        end

        t1+=1
    end
end




function CorrelationGen(n,x)
    P = zeros(Float64,2,2,n,n)
    for v = 1:n, w = 1:n
        if v == w
            P[1,1,v,w] = x/n
            P[2,1,v,w] = 0
            P[1,2,v,w] = 0
            P[2,2,v,w] = 1 - x/n
        else
            P[1,1,v,w] = (x*(x-1))/(n*(n-1))
            P[2,1,v,w] = x/n - x*(x-1)/(n*(n-1))
            P[1,2,v,w] = x/n - x*(x-1)/(n*(n-1))
            P[2,2,v,w] = 1 - 2*x/n + x*(x-1)/(n*(n-1))
        end
    end
    return P
end

function find_d(P)
    d = 1
    while d<10 #loop looking for dimension
        c = gradientdescent(P,d, alpha=0.3, beta=0.8)
        if c < 1e-5
            println("Dimension is less or equal then $d")
            return d
        end

        d+=1
    end
    print("NO DIMENSION LESS THEN 10 WAS FOUND")
end

function finddims(n,m)
    x = zeros(m)
    Pcor = zeros(ComplexF64,m,2,2,n,n)

    x[1] = 1 + 1 /(n-1)
    Pcor[1,:,:,:,:] = CorrelationGen(n,x[1])

    for i in 2:(m)
        x[i] = 1 + 1 / (n - 1 - x[i-1])
        Pcor[i,:,:,:,:] = CorrelationGen(n,x[i])
    end
    println(x)

    d = zeros(Int64,m)
    for i in 3:m
        d[i] = find_d(Pcor[i,:,:,:,:] )
    end
    for i = 1:m
        println("dimension of x_$i = $(d[i])")
    end

end

function f(P,X,Y,psi)
    Pa= size(X,3)
    Pb= size(Y,3)
    Ps= size(X,4)
    Pt= size(Y,4)
    d= size(X,1)
    E = zeros(ComplexF64,d,d,Pa,Ps) #matrix of matrici
    F = zeros(ComplexF64,d,d,Pb,Pt)
    for a = 1:Pa, s = 1:Ps
        E[:,:,a,s] = adjoint(X[:,:,a,s])*X[:,:,a,s]
    end
    for b = 1:Pb, t = 1:Pt
        F[:,:,b,t] = adjoint(Y[:,:,b,t])*Y[:,:,b,t]
    end
    cost(P,E,F,psi)[1]
end

function directionalderivative(P,f, X, Y, psi, dX, dY, dpsi)
    ns = sqrt(gradientsquare(dX, dY, dpsi))
    dXtest = dX/ns
    dYtest = dY/ns
    dpsitest = dpsi/ns
    #println(" gradientsquare $(gradientsquare(dXtest, dYtest, dpsitest))")
    eps = 1e-7
    (f(P,X + eps*dXtest, Y + eps*dYtest, psi + eps*dpsitest) - f(P,X, Y, psi)) / eps
end

function test(P,f, X, Y, psi, dX, dY, dpsi)
    steep = directionalderivative(P,f, X, Y, psi, dX, dY, dpsi)
    println("steep: $steep")
    for i1 = 1:d, j1 =1:i1, a=1:Pa, s=1:Ps
        println("|")
        for k1 = 1:2, k2 = 1:2
            Xtest .= dX
            Xtest[i1,j1,a,s] = dX[i1,j1,a,s]+real(dX[i1,j1,a,s])*0.1*(-1)^k1 +imag(dX[i1,j1,a,s])*im*0.1*(-1)^k2
            dir = directionalderivative(P,f, X, Y, psi, Xtest, dY, dpsi)
            if  dir > steep
                println("MORE i1: $i1, j1: $j1, a: $a, s: $s, k1: $k1, k2: $k2, dif: $(dir-steep)")
            else
                println("LESS i1: $i1, j1: $j1, a: $a, s: $s, k1: $k1, k2: $k2, dif: $(dir-steep)")
            end
        end
    end
    for  i2 =1:d, j2=1:i2,b=1:Pb, t=1:Pt
        println("|")
        for k1 = 1:2, k2 = 1:2
            Ytest.= dY
            Ytest[i2,j2,b,t] = dY[i2,j2,b,t]+real(dY[i2,j2,b,t])*0.1*(-1)^k1+imag(dY[i2,j2,b,t])*im*(0.1)*(-1)^k2
            dir = directionalderivative(P,f, X, Y, psi, dX, Ytest, dpsi)
            if  dir > steep
                println("MORE i2: $i2, j2: $j2, b: $b, t: $t, k1: $k1, k2: $k2, dif: $(dir-steep)")
            else
                println("LESS i2: $i2, j2: $j2, b: $b, t: $t, k1: $k1, k2: $k2, dif: $(dir-steep)")
            end
        end
    end
    for i3 = 1:d^2
        println("|")
        for k1 = 1:2, k2= 1:2
            psitest .= dpsi
            psitest[i3] = dpsi[i3]+real(dpsi[i3])*(0.1)*(-1)^k1+imag(dpsi[i3])*im*(0.1)(-1)^k2
            dir = directionalderivative(P,f, X, Y, psi, dX, dY, psitest)
            if  dir > steep
                println("MORE i3: $i3, k1: $k1, k2: $k2, dif: $(dir-steep)")
            else
                println("LESS i3: $i3, k1: $k1, k2: $k2, dif: $(dir-steep)")
            end
        end
    end
    steep = directionalderivative(P,f, X, Y, psi, dX, dY, dpsi)
    println("steep end: $steep")
end



function testgrad()
    d =2
    Pa= 2
    Pb= 2
    Ps= 2
    Pt= 2

    psi = randn(ComplexF64,d^2)
    psi /= norm(psi)

    X = zeros(ComplexF64,d,d,Pa,Ps)  #dimensies uit p halen
    Y = zeros(ComplexF64,d,d,Pb,Pt)
    for a = 1:Pa,s =1:Ps,i=1:d, j=1:i
        X[i,j,a,s] = randn(ComplexF64)/d
    end
    for b = 1:Pb,t =1:Pt,i=1:d, j=1:i
        Y[i,j,b,t] = randn(ComplexF64)/d
    end

    dX = zeros(ComplexF64,d,d,Pa,Ps)
    dY = zeros(ComplexF64,d,d,Pb,Pt)
    dpsi = zeros(ComplexF64, d^2)

    Xtest = zeros(ComplexF64,d,d,Pa,Ps)
    Ytest = zeros(ComplexF64,d,d,Pb,Pt)
    psitest = zeros(ComplexF64, d^2)

    E = zeros(ComplexF64,d,d,Pa,Ps) #matrix of matrici
    F = zeros(ComplexF64,d,d,Pb,Pt)

    Etest = zeros(ComplexF64,d,d,Pa,Ps) #matrix of matrici
    Ftest = zeros(ComplexF64,d,d,Pb,Pt)

    gradient!(PCHSH, X, Y, psi, dX, dY, dpsi, d, Pa, Pb, Ps, Pt)
    test(PCHSH,f, X, Y, psi, dX, dY, dpsi)
end



function parameterana(P,d)
    alpha = 0.1:0.1:0.5
    beta = 0.1:0.1:0.9
    timemeas = zeros(Float64, size(alpha,1), size(beta,1))
    for a = 1: size(alpha,1), b = 1:size(beta,1)
        for t = 1:10
            timemeas[a,b] += @elapsed gradientdescent(P,d,alpha = alpha[a], beta = beta[b])
        end
    end
    timemeas /= 5
    return(timemeas)
end

function timetaken()
    P = PcorGen8
    cst = []
    grad = []
    Pa= size(P,1)
    Pb= size(P,2)
    Ps= size(P,3)
    Pt= size(P,4)

    for d = 1:20
        psi = randn(ComplexF64,d^2)
        psi /= norm(psi)

        X = zeros(ComplexF64,d,d,Pa,Ps)  #dimensies uit p halen
        Y = zeros(ComplexF64,d,d,Pb,Pt)
        for a = 1:Pa,s =1:Ps,i=1:d, j=1:i
            X[i,j,a,s] = randn(ComplexF64)/d
        end
        for b = 1:Pb,t =1:Pt,i=1:d, j=1:i
            Y[i,j,b,t] = randn(ComplexF64)/d
        end

        dX = zeros(ComplexF64,d,d,Pa,Ps)
        dY = zeros(ComplexF64,d,d,Pb,Pt)
        dpsi = zeros(ComplexF64, d^2)



        push!(cst , @elapsed f(P,X,Y,psi))
        push!(grad, @elapsed gradient!(P, X, Y, psi, dX, dY, dpsi, d, Pa, Pb, Ps, Pt))
    end
    plot(hcat(cst,grad),xlabel = "dimension", ylabel = "time elapsed (s)",label = ["cost" "gradient"])
end



function C(r)
    M = zeros(r, binomial(r, 2)+1)
    c = 1
    for i = 1:(r-1), j=i+1:r
        M[i, c] = 1/sqrt(2)
        M[j, c] = -1/sqrt(2)
        c += 1
    end
    for j=1:r
        M[j, end] = 1/sqrt(r)
    end
    M
end

function getxandy(M)
    L, U, p = lu(M')
    s = zeros(Int, length(p))
    for i=1:length(p)
        s[p[i]] = i
    end
    U, L[s,:]'
end

function b(i, d)
    v = zeros(d)
    v[i] = 1.0
    v
end

function psi(d)
    1/sqrt(d) * sum(kron(b(i, d), b(i, d)) for i=1:d)
end

function phi(r, i)
    X = [0.0 1.0; 1.0 0.0]
    Y = [0.0 -im; im 0]
    Z = [1.0 0.0; 0.0 -1.0]
    if isodd(i)
        A = reduce(kron, [Z for _ = 1:div(i-1, 2)], init=ones(1,1))
        A = kron(A, X)
        s = div(r+1,2) - div(i+1,2)
        A = kron(A, Matrix(I,2^s,2^s))
        A
    else
        A = reduce(kron, [Z for _ = 1:div(i-2, 2)], init=ones(1,1))
        A = kron(A, Y)
        s = div(r+1,2) - div(i,2)
        A = kron(A, Matrix(I,2^s,2^s))
        A
    end
end

function Xmat(v)
    sum(v[i] * phi(length(v), i) for i=1:length(v))
end

function Ymat(v)
    sum(v[i] * transpose(phi(length(v), i)) for i=1:length(v))
end

function getcliffordrep(r)
    M = C(r)
    d = 2^div(r+1,2)
    mypsi = psi(d)
    X, Y = getxandy(M)
    mypsi, [Xmat(X[:,i]) for i=1:size(X,2)], [Ymat(Y[:,i]) for i=1:size(Y,2)]
end

function obsertoPOVM(X)
    r = size(X,1)
    d = size(X[1],1)
    XP = zeros(ComplexF64,d,d,2,r)
    for s = 1:size(X,1)
        for a = 1:2
            XP[:,:,a,s]  = (I + (-1)^a*X[s])/2
        end
    end
    XP
end

function generatePGD(r)
    mpsi, X,Y = getcliffordrep(r)
    XPOVM = obsertoPOVM(X)
    YPOVM = obsertoPOVM(Y)
    PGD = Pcalc(XPOVM,YPOVM,mpsi)
end
PCHSH = Pcalc(CHSH()...)

PGD1 = generatePGD(1)
PGD2 = generatePGD(2)
PGD3 = generatePGD(3)
PGD4 = generatePGD(4)
PGD5 = generatePGD(5)
PGD6 = generatePGD(6)

PcorGen3= CorrelationGen(3,3.0/2)
PcorGen4= CorrelationGen(4,(4.0/3.0))
PcorGen4_2= CorrelationGen(4,(8.0/5.0))
PcorGen4_3= CorrelationGen(4,(12.0/7.0))
PcorGen5= CorrelationGen(5,(5.0/4.0))
PcorGen6 = CorrelationGen(6,6.0/5)
PcorGen8 = CorrelationGen(8,8.0/7)



#
# gradientdescentPVM(PcorGen4_2,5)
# 
# gradientdescent(PcorGen4_2,9)
#
# gradientdescentNOLINE(PCHSH,2)#nope
# gradientdescent(PCHSH,2)
# gradientdescentMomentum(PCHSH,2)
# gradientdescentNAG(PCHSH,2,t = 1/30, gamma = 0.6)
#
# gradientdescent(PcorGen4_2,5)
#
#
# gradientdescent(PcorGen4_2,9)
# gradientdescent(PcorGen4_2,27) #300 0.016
# gradientdescent(PcorGen4_2,29) # todo
#
# gradientdescent(PCHSH,2, alpha = .5, beta = .5)
# gradientdescentNOLINE(PCHSH,2,t=20)
#
# gradientdescent(PcorGen4_3,7)
# gradientdescent(PcorGen4_3,9)
# gradientdescent(PcorGen4_3,11)
# gradientdescent(PcorGen4_3,11)
# gradientdescent(PcorGen4_3,13)
# gradientdescent(PcorGen4_3,15)
#
# gradientdescentNOPEN(PcorGen4,3)
# #
# # gradientdescentNAG(PcorGen6,5,eta = 1/130) #3??? niet verwacht gaat wel echt naar nul
# #
# # gradientdescent(PcorGen4,3) #3??? niet verwacht gaat wel echt naar nul
# gradientdescent(PcorGen8,5)
# # @profiler gradientdescent(PcorGen6,4)
# # @profiler gradientdescent(PcorGen6,5)
# # gradientdescent(PcorGen6,6)
# #
# # gradientdescent(PcorGen8,6)
# gradientdescent(PcorGen8,7)
# gradientdescent(PcorGen8,8)
# # #
# # #
# # # gradientdescentNOLINE(PCHSH,4)
# # # gradientdescentNOLINE(PcorGen4,3)
# # #
# # #
# # gradientdescentNOLINE(PcorGen3,3)
# # # find_d(PCHSH)
# # find_d(PcorGen) #3!??!
#
#
#
#
#

# gradientdescent(PGD1,2)

# @elapsed gradientdescent(PGD2,2)

# gradientdescent(PGD3,2)

# gradientdescentNOLINE(PGD4,4,t= 1/70)
# gradientdescent(PGD4,4,beta = .5)
# gradientdescentMomentum(PGD4, 4, t = 1/30, gamma = .9)
# gradientdescentPVM(PGD4,4,beta= .7, alpha = .3)

# gradientdescent(PGD5,3)

# gradientdescentMomentum(PGD6,8, t = 1/70)
#
# r=4
# mpsi, X,Y = getcliffordrep(r)
# XPOVM = obsertoPOVM(X)
# YPOVM = obsertoPOVM(Y)
# PGD4 =Pcalc(XPOVM,YPOVM,mpsi)
# gradientdescent(PGD4,4)
