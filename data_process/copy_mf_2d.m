% GFGU MFDMA 1D.m
function [ n , Fq , tau , alpha , f ] = GFGU MFDMA 1D( x , n min , n max , N, t h e t a , q )
%
% The p r o c e d u r e [ n , Fq , tau , alpha , f ]=GFGU MFDMA 1D( x , n min , n max ,N, t h e t a , q ) i s
% u sed t o c a l c u l a t e t h e m u l t i f r a c t a l p r o p e r t i e s o f one−d i m e n s i o n a l t ime s e r i e s .
%
% Input :
%
x : t h e t ime s e r i e s we c o n s i d e r e d
%
n min : t h e l o w e r bound o f t h e segment s i z e n
%
n max : t h e u pper bound o f t h e segment s i z e n
%
N: t h e l e n g t h o f n , t h a t i s , t h e d a t a p o i n t s i n t h e p l o t o f Fq VS n9
%
%
%
%
%
%
%
%
%
%
%
%
%
%
%
%
%
%
%
%
%
%
%
%
%
%
%
%
%
%
%
%
t h e t a : t h e p o s i t i o n paramet er o f t h e moving window
q : m u l t i f r a c t a l order
Output :
n : segment s i z e s e r i e s
Fq : q−t h o r d e r f l u c t u a t i o n f u n c t i o n
tau : m u l t i f r a c t a l s c a l i n g exponent
alpha : m u l t i f r a c t a l s i n g u l a r i t y strength function
f : m u l t i f r a c t a l spect ru m
The p r o c e d u r e works as f o l l o w s :
1) C o n s t r u c t t h e c u m u l a t i v e sum y .
2) For each n , c a l c u l a t e t h e moving a v e r a g e f u n c t i o n \ w i d e t i l d e { y } .
3) Determine t h e r e s i d u a l e by d e t r e n d i n g \ w i d e t i l d e { y } from y .
4) E s t i m a t e t h e r o o t −mean−s q u a r e f u n c t i o n F .
5) C a l c u l a t e t h e q−t h o r d e r o v e r a l l f l u c t u a t i o n f u n c t i o n Fq .
6) C a l c u l a t e t h e m u l t i f r a c t a l s c a l i n g e x p o n e n t t a u ( q ) .
7) C a l c u l a t e t h e s i n g u l a r i t y s t r e n g t h f u n c t i o n a l p h a ( q ) and spect ru m f ( a l p h a ) .
Note :
1) The window s i z e and t h e segment s i z e must be i d e n t i c a l .
2) The l o w e r bound n min would b e t t e r be s e l e c t e d around 1 0 .
3) The u pper bound n max would b e t t e r be s e l e c t e d around 10% o f t h e l e n g t h o f
t ime s e r i e s .
4) N would b e t t e r be s e l e c e t e d i n t h e ran ge [ 2 0 , 4 0 ] .
5) The paramet er t h e t a v a r i e s i n t h e ran ge [ 0 , 1 ] . Theta = 0 c o r r e s p o n d s t o
backward MFDMA, and t h e t a = 0 . 5 c o r r e s p o n d s t o t h e c e n t e r e d MFDMA, and
t h e t a = 1 c o r r e s p o n d s t o t h e f o r w a r d MFDMA. We recommend t h e t a =0.
Example :
[ n , Fq , tau , alpha , f ]=GFGU MFDMA 1D( x , 1 0 , round ( l e n g t h ( x ) /10) , 3 0 , 0 , − 4 : 0 . 1 : 4 ) ;
i f s i z e ( x , 2 ) == 1
x = x’;
end
M=
MIN
MAX
n =
length ( x ) ;
= log10 ( n min ) ;
= log10 ( n max ) ;
( unique ( round ( logspace (MIN,MAX,N) ) ) ) ’ ;
% C o n s t r u c t t h e c u m u l a t i v e sum y
y = cumsum( x ) ;
for i = 1 : length ( n )
lgth = n( i ,1) ;
% C a l c u l a t e t h e moving a v e r a g e f u n c t i o n \ w i d e t i l d e { y }
y1 = zeros ( 1 ,M −l g t h +1) ;
for j = 1 :M −l g t h +1
y1 ( j ) = mean( y ( j : j+l g t h −1) ) ;
end
% Determine t h e r e s i d u a l e
e=y (max( 1 , f l o o r ( l g t h ∗(1− t h e t a ) ) ) :max( 1 , f l o o r ( l g t h ∗(1− t h e t a ) ) )+length ( y1 ) −1)−y1 ;
% E s t i m a t e t h e r o o t −mean−s q u a r e f u n c t i o n F10
for k=1: f l o o r ( length ( e ) / l g t h )
F{ i } ( k )=sqrt (mean( e ( ( k−1)∗ l g t h +1:k∗ l g t h ) . ˆ 2 ) ) ;
end
end
% C a l c u l a t e t h e q−t h o r d e r o v e r a l l f l u c t u a t i o n f u n c t i o n Fq
for i =1: length ( q )
for j =1: length (F)
f=F{ j } ;
i f q ( i ) == 0
Fq ( j , i )=exp ( 0 . 5 ∗mean( log ( f . ˆ 2 ) ) ) ;
else
Fq ( j , i ) =(mean( f . ˆ q ( i ) ) ) ˆ (1 / q ( i ) ) ;
end
end
end
% Calcu lat e the m u l t i f r a c t a l s c a l i n g exponent tau ( q )
for i =1: s i z e ( Fq , 2 )
f q=Fq ( : , i ) ;
r=r e g s t a t s ( log ( f q ) , log ( n ) , ’ l i n e a r ’ , { ’ t s t a t ’ } ) ;
k=r . t s t a t . beta ( 2 ) ;
h ( i , 1 )=k ;
end
tau=h . ∗ q ’ − 1 ;
% C a l c u l a t e t h e s i n g u l a r i t y s t r e n g t h f u n c t i o n a l p h a ( q ) and spect ru m f ( a l p h a )
dx =7;
dx=f i x ( ( dx−1) / 2 ) ;
for i=dx +1: length ( tau )−dx
xx=q ( i −dx : i+dx ) ;
yy=tau ( i −dx : i+dx ) ;
r=r e g s t a t s ( yy , xx , ’ l i n e a r ’ , { ’ t s t a t ’ } ) ;
a lpha ( i , 1 )=r . t s t a t . beta ( 2 ) ;
end
a lpha=a lpha ( dx+1:end) ;
f=q ( dx +1:end−dx ) ’ . ∗ alpha−tau ( dx+1:end−dx ) ;





% GFGU MFDMA 2D.m
function [ n , Fq , tau , alpha , f ] = GFGU_MFDMA_2D(X, n_min , n_max , N, theta , q )
%
% The procedure GFGU_MFDMA_2D
% is used to calculate the multifractal properties of two−dimensional
% multifractal measures.
%
% Input :
% X: the two−dimensional multifractal measures we considered.
% n_min: the lower bound of the segment size n
% n_max: the upper bound of the segment size n
% N: the length of n , that is , the data points in the plot of Fq VS n
% theta: the position parameter of the moving window
% q: multifractal order
% 
% Output :
% n: segment size series
% Fq: q−th order fluctuation function
% tau: multifractal scaling exponent
% alpha: multifractal singularity strength function
% f: multifractal spectrum
% 
% The procedure works as follows:
% 1) For each n , c o n s t r u c t t h e c u m u l a t i v e sum Y i n a moving window .
% 2) C a l c u l a t e t h e moving a v e r a g e f u n c t i o n \ w i d e t i l d e {Y} .
% 3) Determine t h e r e s i d u a l e by d e t r e n d i n g \ w i d e t i l d e {Y} from Y.
% 4) E s t i m a t e t h e r o o t −mean−s q u a r e f u n c t i o n F .
% 5) C a l c u l a t e t h e q−t h o r d e r o v e r a l l f l u c t u a t i o n f u n c t i o n Fq .
% 6) C a l c u l a t e t h e m u l t i f r a c t a l s c a l i n g e x p o n e n t t a u ( q ) .
% 7) C a l c u l a t e t h e s i n g u l a r i t y s t r e n g t h f u n c t i o n a l p h a ( q ) and spect ru m f ( a l p h a ) .
% 
% Note :
% 1) The window s i z e and t h e segment s i z e must be i d e n t i c a l .
% 2) The l o w e r bound n min would b e t t e r be s e l e c t e d around 1 0 .
% 3) The u pper bound n max would b e t t e r be s e l e c t e d around 10% o f min ( s i z e (X) ) .
% 4) N would b e t t e r be s e l e c e t e d i n t h e ran ge [ 2 0 , 4 0 ] .
% 5) The paramet er t h e t a v a r i e s i n t h e ran ge [ 0 , 1 ] , and we have
% t h e t a=t h e t a 1=t h e t a 2 . Theta = 0 c o r r e s p o n d s t o backward MFDMA, and
% t h e t a = 0 . 5 c o r r e s p o n d s t o t h e c e n t e r e d MFDMA, and t h e t a = 1 c o r r e s p o n d s t o
% t h e f o r w a r d MFDMA. We recommend t h e t a =0.
% 6) In t h e procedu re , we have n=n 1=n 2 f o r t h e segment s i z e .
% 
% Example :
% [ n , Fq , tau , alpha , f ]=GFGU MFDMA 2D(X, 1 0 , round ( min ( s i z e (X) ) /10) , 3 0 , 0 , − 4 : 0 . 1 : 4 ) ;
% 

N1=size(X, 1 ) ;
N2=size(X, 2 ) ;
MIN=log10( n_min ) ;
MAX=log10( n_max ) ;
n=( unique( round( logspace(MIN,MAX,N) ) ) ) ’ ;

for i =1: length( n )
	lgth=n( i , 1 ) ;
	    
	Y = zeros(N1−lgth +1 ,N2−lgth +1) ;
	Y1 = zeros(N1−lgth +1 ,N2−lgth +1) ;
	for j = 1 : N1−lgth +1
		for k = 1 : N2−l g t h +1
			Z = X( j : j+lgth −1 , k : k+lgth −1) ;
			Z1 = (cumsum((cumsum(Z) ) ’ ) ) ’ ;
			% Construct the cumulative sum Y
			Y( j , k )=Z1( end , end) ;
			% Calculate the moving average function \ wide tilde {Y}
			Y1( j , k )=mean( Z1 ( : ) ) ;
		end
	end
	
	% Determine the residual e
	x0 =1: size(Y, 1 )−min( floor( lgth ∗ theta ) , lgth −1) ;
	y0 =1: size(Y, 2 )−min( floor( lgth ∗ theta ) , lgth −1) ;
	x1=size (Y1 , 1 )−length ( x0 ) +1: size (Y1 , 1 ) ;
	y1=size (Y1 , 2 )−length ( y0 ) +1: size (Y1 , 2 ) ;
	e=Y( x0 , y0 )−Y1( x1 , y1 ) ;

	% Estimate the root−mean−square function F
	for k1 =1: floor( size ( e , 1 ) / lgth )
		for k2 =1: floor( size( e , 2 ) / lgth )
			E=e ( ( k1 −1)∗ lgth +1: k1 ∗ lgth , ( k2 −1)∗ lgth +1: k2 ∗ lgth ) ;
			F{ i } ( k1 , k2 )=sqrt(mean(E ( : ). ˆ2 ) ) ;
		end
	end
end

% Calculate the q−th order overall fluctuation function Fq
for i =1: length( q )
	for j =1: length(F)
		f=F{ j } ( : ) ;
		if q( i ) == 0
			Fq( j , i )=exp( 0.5 ∗mean( log( f. ˆ2 ) ) ) ;
		else
			Fq( j , i ) =(mean( f. ˆq ( i ) ) ) ˆ (1 / q ( i ) ) ;
		end
	end
end

% Calculate the multifractal scaling exponent tau( q )
for i =1: size( Fq , 2 )
	fq=Fq( : , i ) ;
	r=regstats( log( fq ) , log( n ) , ’linear’ , { ’tstat’ } ) ;
	k=r . tstat . beta( 2 ) ;
	h( i , 1 )=k ;
end
tau=h. ∗ q’ − 2 ;

% Calculate the singularity strength function alpha( q ) and spectrum f( alpha )
dx =7;
dx=fix ( ( dx−1) / 2 ) ;
for i=dx +1: length( tau )−dx
	xx=q( i −dx : i+dx ) ;
	yy=tau( i −dx : i+dx ) ;
	r=regstats( yy , xx , ’linear’ , { ’tstat’ } ) ;
	alpha( i , 1 )=r . tstat . beta( 2 ) ;
end
alpha=alpha( dx+1:end) ;
f=q( dx +1:end−dx )’. ∗ alpha−tau (dx+1:end−dx) ;


