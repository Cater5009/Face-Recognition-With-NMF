function [H,grad,iter] = nlssubprob(V,W,Hinit,tol,maxiter)

H = Hinit; 
WtV = W' * V; 
WtW = W' * W; 

alpha = 1; 
beta = 0.1;
for iter = 1 : maxiter
    grad = WtW * H - WtV;
    projgrad = norm(grad(grad < 0 | H >0));
    if projgrad < tol
        break
    end
    
    %ËÑË÷²½³¤
    for inner_iter = 1 : 20
        Hn = max(H - alpha * grad, 0);
        d = Hn - H;
        gradd = sum(sum(grad.* d));
        dQd = sum(sum((WtW * d).* d));
        suff_decr = 0.99 * gradd + 0.5 * dQd < 0;
        if inner_iter == 1
            decr_alpha = ~suff_decr;
            Hp = H;
        end
        if decr_alpha
            if suff_decr
                H = Hn;
                break;
            else
                alpha = alpha * beta;
            end
        else
            if ~suff_decr | Hp == Hn
                H = Hp;
                break;
            else
                alpha = alpha/beta;
                Hp = Hn;
            end
        end
    end
end

if iter==maxiter
    fprintf('Max iter in nlssubprob\n');
end