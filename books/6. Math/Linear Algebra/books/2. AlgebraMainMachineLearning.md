
MATRIZ: (100,10)
- 100 rows
- 10 features

Aqui é obrigatório termos 10 pesos + bias, como é por iteração, cada batch vai ser = [a,b,c,d,e,f,g,h, i, j]	* [wa, wb, wc, wd, we, wf, wg, wh, wi, wj] 												    		1 SAMPLE             	WEIGHTS

Então aqui teríamos que ter (1, 10) para o SAMPLE e (1, 10), mas como sabemos daria erro, então precisaríamos transpor os pesos para virar [ [wa],
																	     [wb],
																	     [wc],
																	     ...
																		 ]
Para assim obter a conta, e teríamos z = ([wa * a] + [wb + b] [wc + c]), se tivéssemos (400, 10), precisaríamos ter (10,1) e assim geraria um Z de (400,1)
Depois teríamos que calcular a loss, e para isso teríamos o Z = (400,1) e precisaríamos dos targets desses que seriam ainda no dataset (400, 1).
Como é uma subtração agora, podemos manter em suas formas originais Z(400, 1) e Y(400,1) pq ai iriamos subtrair mesmo e obteríamos a L(400,1),  e ai faríamos a média desses 400, e teríamos a LOSS do batch (1,1)