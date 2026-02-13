***Компактная разностная схема***

$A_x\circ\frac{\Delta}{\tau}\circ\mathbfit{u}_h + A_t\circ\frac{\Delta}{h}\circ\mathbfit{f}(\mathbfit{u}_h) = 0$

$A_x = \frac{T_h + 4E+T_{-h}}{6}\:\:\: A_t = \frac{T^{\tau} + 4E + T^{-\tau}}{6}$

$\Delta_x=\frac{T_h-T_{-h}}{2}\:\:\:\Delta_t = \frac{T^{}\tau-T^{}-\tau}{2}$

***Стабилизация компактной схемы***

*Стабилизация проводится путем введния искусственной вязкости четвертого порядка дивергентности*

$(\mathbfit{w}_h)_i = -C\frac{h^4}{\tau}(\frac{\overline{\Delta}_x}{h})^{4}\circ T^{i\tau} \circ \mathbfit{u}_h\:\:\: i = -1,0,1; \:\:\:\ C_i > 0$


$\overline{\Delta}^{4}_x = (T_{h/2} - T_{-h/2})^4 = T_{2h} - 4T_h + 6E-4T_{-h} + T_{2h}$


$A_x\circ\frac{\Delta}{\tau}\circ\mathbfit{u}_h + A_t\circ\frac{\Delta}{h}\circ\mathbfit{f}(\mathbfit{u}_h) + \frac{h^4}{\tau}\sum_{i=-1}^1 C_i (\frac{\overline{\Delta}_x}{h})^4 \circ T^{i\tau} \circ \mathbfit{u}_h = 0$


\________________________________________________________________________________________

***Алгоритм компактной схемы 3-го порядка для уравнений мелкой воды***

$\overline{U}_t + \overline{F}(U)_x = 0$

$\overline{U} = \left(\begin{smallmatrix} h \\ q \end{smallmatrix}\right)$
$\overline{F}(U) = \left(\begin{smallmatrix} q \\ qu + g\frac{h^2}{2} \end{smallmatrix}\right)$


*Применив операторы, получим:*

$h^{n+1, k+1}_{j+1}(1 + r*u^{n+1, k}_{j+1}) + 4h^{n+1, k+1}_{j} + h^{n+1, k+1}_{j-1}(1 - r*u^{n+1, k}_{j-1}) = h^{n-1}_{j+1} + 4h^{n-1}_{j} + h^{n-1}_{j-1} - r(4(q^n_{j+1} - q^n_{j-1}) + (q^{n-1}_{j+1} - q^{n-1}_{j-1})) \:\:\:, r = \frac{\tau}{h}$

$q^{n+1, k+1}_{j+1}(1 + r*u^{n+1, k}_{j+1}) + 4q^{n+1, k+1}_{j} + h^{n+1, k+1}_{j-1}(1 - r*u^{n+1, k}_{j+1}) = q^{n-1}_{j+1} + 4q^{n-1}_{j} + q^{n-1}_{j-1} -r(4(q^n_{j+1}u^n_{j+1} + g\frac{({h^2})^n_{j+1}}{2} - q^n_{j-1}u^n_{j-1} - g\frac{({h^2})^n_{j-1}}{2}) +     q^{n-1}_{j+1}u^{n-1}_{j+1} + g\frac{({h^2})^{n-1}_{j+1}}{2} - q^{n-1}_{j-1}u^{n-1}_{j-1} - g\frac{({h^2})^{n-1}_{j-1}}{2}) -$ $ -\frac{rg}{2}((h^2)^{n+1, k+1}_{j+1}+ (h^2)^{n+1, k+1}_{j}+ (h^2)^{n+1, k+1}_{j-1})$ - перенесла слагаемые с $h^{n+1, k+1}$ в правую часть, наверное не надо было((

\_________________________________________________________________________________________


$\begin{cases}
  1 + r*u^{n+1, k}_{j+1} - \alpha_{-} \\
  4 - \alpha \\
  1 - r*u^{n+1, k}_{j-1} - \alpha_{+}
\end{cases}$
\-  *значения на диагоналях ленточной матрицы, решается методом прогонки*



