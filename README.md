# フーリエ変換を利用した輪郭近似プログラム
これは「フーリエ変換」の根底は「任意の関数は三角関数の無限級数で表せる」という点であることに注目し、画像から輪郭を検出し、フーリエ変換による形状近似を行う技術への応用が可能なのかについて考えるため作成したプログラムです.

## 実装方針と理論
このプログラムは以下の方針で作成されています。
1. 画像の輪郭を検出
2. 輪郭に対し任意の間隔で点を作成
3. 点群を離散フーリエ変換
4. フーリエ級数展開により形状近似

1,2の操作として、エッジの検出にはCanny法のライブラリを用いました。
3の操作について, 高速フーリエ変換を実装しました。
まず, 離散フーリエ変換は以下の式で定義されます。

$$X(k)=\sum_{n=0}^{N-1}{x_ne^{-\frac{2\pi nkj}{N}}}\ (k=0,\ 1,2\ldots,\ N-1)$$

今回のプログラムでは、離散フーリエ変換の手法として基数2の時間間引き形の高速フーリエ変換を用いました。
時間間引き形アルゴリズムでは、 $x_n$ を偶数番と奇数番に分けて考えます。

$W_N=e^{-\frac{2\pi j}{N}}$とおくと、定義より

$$X\left(k\right)=\sum_{n=0}^{\frac{N}{2}-1}{x\left(2n\right)W_N^{2nk}}+\sum_{n=0}^{\frac{N}{2}-1}{x\left(2n+1\right)W_N^{(2n+1)k}}$$

ここで, $f_{even}=x\left(2n\right),\ f_{odd}=x(2n+1)$ とすると $W_N^{2nk}=W_{\frac{N}{2}}^{nk}$ より

$$X\left(k\right)=\sum_{n=0}^{\frac{N}{2}-1}{f_{even}W_{\frac{N}{2}}^{nk}}+\sum_{n=0}^{\frac{N}{2}-1}{f_{odd}W_{\frac{N}{2}}^{nk}}$$

$$F_even\ (k)\ =\sum_{n=0}^{\frac{N}{2}-1}{f_{even}W_{\frac{N}{2}}^{nk}} $$

$$F_odd\ (k)\ =\sum_{n=0}^{\frac{N}{2}-1}{f_{odd}W_{\frac{N}{2}}^{nk}}$$

よって、

$$X\left(k\right)=F_{even}\left(k\right)+W_N^kF_{odd}(k)$$

定義よりX(k)は $k=0,\ 1,2\ldots,\ N-1$ であるから $W_N^k=W_N^{k-\frac{N}{2}}$ より

$$X\left(k\right)=F_{even}\left(k\right)+W_N^kF_{odd}\ \ \left(0\le k\le\frac{N}{2}-1\right)$$

$$X\left(k\right)=F_{even}\left(k-\frac{N}{2}\right)-W_N^{k-\frac{N}{2}}F_{odd}\left(k-\frac{N}{2}\right)\ \ \left(\frac{N}{2}\le k\le N-1\right)$$

この結果を用いてfft関数の実装を行いました。

## 結果
入力画像とエッジ検出の出力結果は以下の通りです.

![input](result/sample.png)
![edge](result/edges.jpg)

また, 次数の変化に伴う輪郭検出の精度は, 以下の画像から分かるように, 理論通りの結果となりました.
これらの結果から, 輪郭が連続であり, 次数が大きいフーリエ級数展開をおこなうとき, 輪郭の形状近似は非常に高い精度で行えると考えられます。

![result](result/result_2.png)
![result](result/result_5.png)
![result](result/result_10.png)
![result](result/result_15.png)
![result](result/result_50.png)
![result](result/result_100.png)


## 番外編
せっかくフーリエ級数の輪郭近似プログラムを作成したので不連続点を作成しギブス現象の確認も行ってみました。

![result](result/g1.png)
![result](result/g1_result.png)

![result](result/g2.png)
![result](result/g2_result.png)
