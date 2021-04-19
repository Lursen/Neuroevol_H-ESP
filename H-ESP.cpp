// H-ESP.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <algorithm>

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

// метод для подготовки обучающих данных
bool PrepareTrainingData(double* neuralNetInput)
{
    // coming soon
    bool CorrectOutput = true;
    return CorrectOutput;
}

// метод для вычисления приспосабливаемости
double EvaluateFitness(double error)
{
   // do stuff idk
    return error;
}


// метод для пропуска данных через сеть
double EvaluateNet(double* neuralNetInput, double* neuralNetOutput, double* neuralNetHiddenInputW, double* neuralNetHiddenOutputW, double* neuralNetHiddenBias, bool CorrectOutput)
{
    double neuronValue[hiddenNeuronsN] = {0};
    double sum;
    double error[2] = {0};

    //  скрытый слой
    for (int i = 0; i < hiddenNeuronsN; i++)
    {
        sum = 0;
        for (int j = 0; j < inputN; j++)
        {
            sum += *(neuralNetHiddenInputW + i * inputN + j) * double(neuralNetInput[j]);
        }
        sum += neuralNetHiddenBias[i];
        neuronValue[i] = activationFunction(sum);
    }
    // выходной слой
    for (int i=0; i < outputN; i++)
    {
        sum = 0;
        for (int j = 0; j < hiddenNeuronsN; j++)
        {
            sum += *(neuralNetHiddenOutputW + i * inputN + j) * double(neuronValue[j]);
        }
        sum += neuralNetHiddenBias[i];
        neuralNetOutput[i] = activationFunction(sum);
    }

    // рассчёт ошибки выходного слоя 
    for (int i = 0; i < outputN; i++)
    {
        if (CorrectOutput) //1 0
        {
            error[0] = (neuralNetOutput[0] - 1)*(neuralNetOutput[0] - 1);
            error[1] = neuralNetOutput[1];
        }
        else // 0 1
        {
            error[0] = neuralNetOutput[0] * neuralNetOutput[0];
            error[1] = (neuralNetOutput[1] - 1) * (neuralNetOutput[1] - 1);
        }
    }

    double RMSE = 0;
    //RMSE ERROR
    for (int i = 0; i < outputN; i++)
    {
        RMSE += error[i];
    }
    RMSE = RMSE / 2;

    return EvaluateFitness(RMSE);
}

int main()
{
    srand(static_cast<unsigned int>(time(0)));

//Chromosomes
    double* chromosomeInputW =  new double[subpopulationsN * speciesN * inputN];
    double* chromosomeOutputW = new double[subpopulationsN * speciesN * outputN];
    double* chromosomeBias =    new double[subpopulationsN * speciesN];
    double* chromosomeNeuronFitness = new double[subpopulationsN * speciesN];

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
            *(chromosomeNeuronFitness + i*speciesN + j) = 0;
        }
    }

//Neural Nets (Neural Net level)
    double* NNLneuronValue =            new double[neuralNetN * speciesN];                //значение нейрона
    double* NNLneuralNetInput =         new double[neuralNetN * inputN];                  //входные нейроны
    double* NNLneuralNetHiddenInputW =  new double[neuralNetN * hiddenNeuronsN * inputN]; //входные веса нейронов скрытого слоя
    double* NNLneuralNetHiddenOutputW = new double[neuralNetN * hiddenNeuronsN * outputN];//выходные веса нейронов скрытого слоя
    double* NNLneuralNetHiddenBias =    new double[neuralNetN * hiddenNeuronsN];          //смещения нейронов скрытого слоя
    double* NNLneuralNetOutput =        new double[neuralNetN * outputN];                 //выходные нейроны
    double NNLneuralnetFitness[neuralNetN] = {0};                                         //приспосабливаемость сети  

    //создаём случайные сети для Neural Net Level (L2)
    //Проходим по сетям
    for (int i = 0; i < neuralNetN; i++)
    {
        for (int j = 0; j < hiddenNeuronsN; j++)
        {
            // neuron inputs
            for (int k = 0; k < inputN; k++)
            {
                *(NNLneuralNetHiddenInputW + i * hiddenNeuronsN * inputN + j * inputN + k) =  (double)(rand() / (RAND_MAX));
                // NNinput[i][j][k] = ChromInput[i][j][k]
            }
            // neuron outputs
            for (int k = 0; k < outputN; k++)
            {
                *(NNLneuralNetHiddenOutputW + i * hiddenNeuronsN * inputN + j * outputN + k) = (double)(rand() / (RAND_MAX));
                // NNoutput[i][j][k] = ChromOutput[i][j][k]
            }
            // neuron bias
            *(NNLneuralNetHiddenBias + i * hiddenNeuronsN + j) = (double)(rand() / (RAND_MAX));
            // NNbias[i][j] = TTRbias[j];
        }
    }

// Neural Net (Neural Level / trials)
    double* TRneuronValue =            new double[speciesN];                              //значение нейрона
    double* TRneuralNetInput =         new double[inputN];                                //входные нейроны
    double* TRneuralNetHiddenInputW =  new double[hiddenNeuronsN * inputN];               //входные веса нейронов скрытого слоя
    double* TRneuralNetHiddenOutputW = new double[hiddenNeuronsN * outputN];              //выходные веса нейронов скрытого слоя
    double* TRneuralNetHiddenBias =    new double[hiddenNeuronsN];                        //смещения нейронов скрытого слоя
    double* TRneuralNetOutput =        new double[outputN];                               //выходные нейроны
     
//ESP - algorithm
    double best_fitness = 0;
    double goal_fitness = 100;
    double current_fitness = 0;
    int noImprovements = 0;
    int chosenNeurons[subpopulationsN] = {0};

while (best_fitness < goal_fitness)
{
// Evaluation 
    // Trials (Neuron Level)
    for (int trial = 0; trial < hiddenNeuronsN * 10; trial++)
    {
    //создаём случайную сеть
        //Проходим по подпопуляциям/нейронам в создаваемой сети
        for (int i = 0; i < subpopulationsN; i++)
        {
            int j = rand()%hiddenNeuronsN;
            chosenNeurons[i] = j;
            // neuron inputs
            for (int k = 0; k < inputN; k++)
            {
                *(TRneuralNetHiddenInputW + i * inputN + k) = *(chromosomeInputW + i * speciesN * inputN + j * inputN + k); 
                // NNinput[i][k] = ChromInput[i][j][k]
            }
            // neuron outputs
            for (int k = 0; k < outputN; k++)
            {
                *(TRneuralNetHiddenOutputW + i * outputN + k) = *(chromosomeOutputW + i * speciesN * inputN + j * outputN + k);
                // NNoutput[i][k] = ChromOutput[i][j][k]
            }
            // neuron bias
            TRneuralNetHiddenBias[i] = *(chromosomeBias + i * inputN + j);
            // NNbias[i] = ChromBias[i][j];
        }

        // подготовка и пропуск данных через нейронную сеть
        bool CorrectOutput = PrepareTrainingData(TRneuralNetInput);
        current_fitness = EvaluateNet(TRneuralNetInput, TRneuralNetOutput, TRneuralNetHiddenInputW, TRneuralNetHiddenOutputW, TRneuralNetHiddenBias, CorrectOutput);

        // подсчёт приспосабливаемости
        for (int i = 0; i < subpopulationsN; i++)
        {
            *(chromosomeNeuronFitness + i * speciesN + chosenNeurons[i]) += current_fitness;
            // chromFitness[i][chosenNeurons[i]]
        }

        // сохраняем лучшее значение приспосабливаемости
        if (current_fitness > best_fitness)
        {
            best_fitness = current_fitness;
            noImprovements = 0;
        }
        else
        {
            noImprovements++;
        }

    // сравнение сети с L2
        vector<pair<int, int> > vp;
        vp.clear();
        // сохраняем fitness с индексом после сортировки
        for (int i = 0; i < neuralNetN; ++i)
        {
            vp.push_back(make_pair(NNLneuralnetFitness[i], i));
        }

        sort(vp.begin(), vp.end());

        // если полученная сеть лучше худшей из L2
        if (current_fitness > vp[0].first)
        {
            // заменяем худшую сеть из L2 на полученную
            NNLneuralnetFitness[vp[0].second] = current_fitness;

            for (int j = 0; j < hiddenNeuronsN; j++)
            {
                for (int k = 0; k < inputN; k++)
                {
                    //NNL[vp[0].second][j][k]=TR[j][k]
                    *(NNLneuralNetHiddenInputW + vp[0].second * hiddenNeuronsN * inputN + j * inputN + k) = *(TRneuralNetHiddenInputW + j * inputN + k);
                }
                for (int k = 0; k < outputN; k++)
                {
                    //NNL[vp[0].second][j][k]=TR[j][k]
                    *(NNLneuralNetHiddenOutputW + vp[0].second * hiddenNeuronsN * outputN + j * inputN + k) = *(TRneuralNetHiddenOutputW + j * outputN + k);
                }
                // neuron bias
                *(NNLneuralNetHiddenBias + vp[0].second * hiddenNeuronsN + j) = TRneuralNetHiddenBias[j];
                // NNbias[vp[0].second][j] = TTRbias[j];
            }
        }

     // проверка на стагнацию?
        if (noImprovements > 5)
        {
            // DO STUFF
            // idk if it's necessarry
        }

    } // the end of trials

// Neural Net Level 
    // вычисление приспособленности
    for (int i = 0; i < neuralNetN; i++)
    {
        // подготовка и пропуск данных через нейронную сеть
        bool CorrectOutput = PrepareTrainingData(NNLneuralNetInput);
        NNLneuralnetFitness[i] = EvaluateNet(NNLneuralNetInput, NNLneuralNetOutput, NNLneuralNetHiddenInputW, NNLneuralNetHiddenOutputW, NNLneuralNetHiddenBias, CorrectOutput);
    }


}



}


