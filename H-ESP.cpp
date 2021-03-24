// H-ESP.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

const int hiddenLayersN = 1; //скрытые слои
const int hiddenNeuronsN = 10; //число нейронов в скрытом слое\подпопуляций
const int subpopulationsN = hiddenNeuronsN; //число подпопуляций
const int speciesN = 10; //особей в популяции 
const int neuralNetN = 5; //нейронных сетей 

const int inputN = 9; //число входнных нейронов
const int outputN = 2; //число выходных нейронов 

double activationFunction(double S)
{
    // Функция активации
    return  (1.0 / (1.0 + exp(-S)));
}
double EvaluateNet(double* neuralNetInput, double* neuralNetOutput, double* neuralNetHiddenInputW, double* neuralNetHiddenOutputW, double* neuralNetHiddenBias)
{
    double neuronValue[hiddenNeuronsN];
    //  скрытый слой
    for (int i = 0; i < hiddenNeuronsN; i++)
    {
        double sum = 0;

        for (int j = 0; j < inputN; j++)
        {
            sum += *(neuralNetHiddenInputW + i * inputN + j) * double(neuralNetInput[j]);
        }
        sum += neuralNetHiddenBias[i];
        neuronValue[i] = activationFunction(sum);
    }

}

int main()
{
    srand(static_cast<unsigned int>(time(0)));

//Chromosomes
    double* chromosomeInputW =  new double[subpopulationsN * speciesN * inputN];
    double* chromosomeOutputW = new double[subpopulationsN * speciesN * outputN];
    double* chromosomeBias =    new double[subpopulationsN * speciesN];

    //инициализация хромосом
    //подпопуляции
    for (int i = 0; i < subpopulationsN; i++)
    {
        //особи (нейроны)
        for (int j = 0; j < speciesN; j++)
        {
            //значения хромосом (веса)
            for (int k = 0; k < inputN; k++)
            {
                *(chromosomeInputW + i * speciesN * inputN + j * inputN + k) = ((double)rand() / (RAND_MAX));
            }
            for (int k = 0; k < outputN; k++)
            {
                *(chromosomeOutputW + i * speciesN * inputN + j * inputN + k) = ((double)rand() / (RAND_MAX));
            }
            
            *(chromosomeBias + i*speciesN + j) = ((double)rand() / (RAND_MAX));
        }
    }

//Neural Nets (Neural Net level)
    double* NNLneuronValue =            new double[neuralNetN * speciesN];                //значение нейрона
    double* NNLneuralNetInput =         new double[neuralNetN * inputN];                  //входные нейроны
    double* NNLneuralNetHiddenInputW =  new double[neuralNetN * hiddenNeuronsN * inputN]; //входные веса нейронов скрытого слоя
    double* NNLneuralNetHiddenOutputW = new double[neuralNetN * hiddenNeuronsN * outputN];//выходные веса нейронов скрытого слоя
    double* NNLneuralNetHiddenBias =    new double[neuralNetN * hiddenNeuronsN * outputN];//смещения нейронов скрытого слоя
    double* NNLneuralNetOutput =        new double[neuralNetN * outputN];                 //выходные нейроны

// Neural Net (trials)
    double* TRneuronValue =            new double[speciesN];                              //значение нейрона
    double* TRneuralNetInput =         new double[inputN];                                //входные нейроны
    double* TRneuralNetHiddenInputW =  new double[hiddenNeuronsN * inputN];               //входные веса нейронов скрытого слоя
    double* TRneuralNetHiddenOutputW = new double[hiddenNeuronsN * outputN];              //выходные веса нейронов скрытого слоя
    double* TRneuralNetHiddenBias =    new double[hiddenNeuronsN * outputN];              //смещения нейронов скрытого слоя
    double* TRneuralNetOutput =        new double[outputN];                               //выходные нейроны

// Trials (Neuron Level)
    // по всем подпопуляциям
    for (int i = 0; i < subpopulationsN; i++)
    {

    }

// Neural Net Level - инициализация нейронных сетей (выбор случайных нейронов)
    for (int i = 0; i < neuralNetN; i++)
    {

        
    }
}

