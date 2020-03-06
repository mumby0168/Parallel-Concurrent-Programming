#include <stdio.h>


void multiplyMatrices()
{
	const int heightA = 4;
	const int widthA = 3;

	const int heightB = 3;
	const int widthB = 2;

	const int arraySizeA = heightA * widthA;
	const int arraySizeB = heightB * widthB;
	const int arraySizeC = heightA * widthB;

	const int matrixA[arraySizeA] = { 10,5,4, 
									  3,2,5,
									  6,7,8,
									  9,7,7};

	const int matrixB[arraySizeB] = { 10, 5,
								      4, 3, 
		                              2, 5 };
	int result[arraySizeC] = {};

	for (int i = 0; i < widthA; i++)
	{
		printf("i: %d", i);
		int sum = 0;
		for (int j = 0; j < widthA * i; j++)
		{
			printf("j: %d", j);
		}
	}



}