#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <algorithm>
#include <fstream>
#include <random>

using namespace std;

int hiddenNeuronsN = 50;					//число нейронов в скрытом слое
int subpopulationsN = hiddenNeuronsN;           //число подпопуляций
int speciesN = 10;						//особей в популяции

const int hiddenLayersN = 1; 					//скрытые слои
const int inputN = 9;                           //число входнных нейронов
const int outputN = 2;                          //число выходных нейронов
//длина генотипа
const int gene_lenght = inputN + outputN + 1;

const double mutRate = 0.6; // частота мутаций

// информация о датасете
const int dataset_rows = 699; // кол-во строк датасета
const int col = 11; // кол-во столбцов датасета
const double coef = 0.8; // процент данных для тренировочной выборки

// распределение Коши 
cauchy_distribution<double> cauchy_distr(0.0, 1);
uniform_real_distribution<double> uniform_distr(-1.0, 1.0);

default_random_engine generator;

// метод для подготовки обучающих данных
bool PrepareTrainingData(vector<vector<double>>& dataset, vector<double>& neuralNetInput)
{
    int rnd = rand() % (dataset.size()-1);

    for (int i = 0; i < inputN; i++)
    {
        neuralNetInput[i] = dataset[rnd][i];
    }

    bool CorrectOutput = dataset[rnd][inputN + 1];
    return CorrectOutput;
}

double activationFunction(double S)
{
    // Функция активации
    return  (1.0 / (1.0 + exp(-S)));
}

double CauchyDistr()
{
    double f = uniform_distr(generator);
    //double f = ((double)rand() / (RAND_MAX)) - 1.0;
    return f;
}

void mutate(vector<vector<vector<double>>>& chromosomeWeights, int subPopulation, int neuron)
{
    double rnd = ((double)rand() / (RAND_MAX));

    if (rnd < mutRate)
    {
        int gene = rand() % gene_lenght;
        chromosomeWeights[subPopulation][neuron][gene] += CauchyDistr();
    }
}

void crossoverNeuronLevel(vector<vector<vector<double>>>& Weights, int subPopulation, int fNeuron, int sNeuron, int fdNeuron, int sdNeuron)
{
    // точка разрыва между генами
    int crosspoint = rand() % (gene_lenght);

    for (int k = 0; k < gene_lenght; k++)
    {
        if (k < crosspoint)
        {
            //значения генов (веса)
            Weights [subPopulation ][fdNeuron][k] = Weights[subPopulation][fNeuron][k];
        }
        else
        {
            //значения генов (веса)            
            Weights[subPopulation][sdNeuron][k] = Weights[subPopulation][sNeuron][k];
        }
    }
}

void BurstMutate(vector<vector<vector<double>>>& chromosomeWeights, vector<vector<double>>& chromosomeNeuronFitness)
{
    vector<pair<double, int>> vp;

    for (int i = 0; i < subpopulationsN; i++)
    {
        vp.clear();

        for (int j = 0; j < speciesN; j++)
        {
            vp.push_back(make_pair(chromosomeNeuronFitness[i][j], j));
        }

        sort(vp.rbegin(), vp.rend());

        for (int j = 0; j < speciesN; j++)
        {
            for (int k = 0; k < gene_lenght; k++)
            {
                chromosomeWeights[i][j][k] = chromosomeWeights[i][vp[0].second][k] + CauchyDistr();
            }
        }
    }
}

// метод для пропуска данных через сеть
double EvaluateNet(vector<double>& Input, vector<vector<double>>& InputW, vector<vector<double>>& OutputW, vector<double>& Bias, bool CorrectOutput)
{
    double sum;
    double error[outputN] = { 0 };
    double neuralNetOutput[outputN] = { 0 };
    vector<double> neuronValue(hiddenNeuronsN, 0);

    //  скрытый слой
    for (int i = 0; i < hiddenNeuronsN; i++)
    {
        sum = 0;
        for (int j = 0; j < inputN; j++)
        {
            sum += InputW[i][j] * double(Input[j]);
        }
        sum += Bias[i];
        neuronValue[i] = activationFunction(sum);
    }
    // выходной слой
    for (int i = 0; i < outputN; i++)
    {
        sum = 0;
        for (int j = 0; j < hiddenNeuronsN; j++)
        {
            sum += OutputW[j][i] * double(neuronValue[j]);
        }
        neuralNetOutput[i] = activationFunction(sum);
    }

    // рассчёт ошибки выходного слоя
    if (CorrectOutput) //1 0
    {
        error[0] = (neuralNetOutput[0] - 1) * (neuralNetOutput[0] - 1);
        error[1] = neuralNetOutput[1] * neuralNetOutput[1];
    }
    else // 0 1
    {
        error[0] = neuralNetOutput[0] * neuralNetOutput[0];
        error[1] = (neuralNetOutput[1] - 1) * (neuralNetOutput[1] - 1);
    }

    double MSE = 0;
    //RMSE ERROR
    for (int i = 0; i < outputN; i++)
    {
        MSE += error[i];
    }
    MSE = (MSE / outputN);

    // вычисление приспосабливаемости
    return  1.0 - MSE;
}

// метод для пропуска данных через сеть
double EvaluateLesionedNet(vector<double>& Input, vector<vector<double>>& InputW, vector<vector<double>>& OutputW, vector<double>& Bias, bool CorrectOutput, int lesionedNeuron)
{
    double sum;
    double error[outputN] = { 0 };
    double neuralNetOutput[outputN] = { 0 };
    vector<double> neuronValue(hiddenNeuronsN, 0);

    //  скрытый слой
    for (int i = 0; i < hiddenNeuronsN; i++)
    {
        if (i != lesionedNeuron)
        {
            sum = 0;
            for (int j = 0; j < inputN; j++)
            {
                sum += InputW[i][j] * double(Input[j]);
            }
            sum += Bias[i];
            neuronValue[i] = activationFunction(sum);
        }
    }
    // выходной слой
    for (int i = 0; i < outputN; i++)
    {
        sum = 0;
        for (int j = 0; j < hiddenNeuronsN; j++)
        {
            if (j != lesionedNeuron)
            {
                sum += OutputW[j][i] * double(neuronValue[j]);
            }
        }
        neuralNetOutput[i] = activationFunction(sum);
    }

    // рассчёт ошибки выходного слоя
    if (CorrectOutput) //1 0
    {
        error[0] = (neuralNetOutput[0] - 1) * (neuralNetOutput[0] - 1);
        error[1] = neuralNetOutput[1] * neuralNetOutput[1];
    }
    else // 0 1
    {
        error[0] = neuralNetOutput[0] * neuralNetOutput[0];
        error[1] = (neuralNetOutput[1] - 1) * (neuralNetOutput[1] - 1);
    }

    double MSE = 0;
    //RMSE ERROR
    for (int i = 0; i < outputN; i++)
    {
        MSE += error[i];
    }
    MSE = (MSE / outputN);

    // вычисление приспосабливаемости
    return  1.0 - MSE;
}

void GetBestNetwork( vector<vector<double>>& chromosomeNormNeuronFitness, vector<vector<double>>& InputW, vector<vector<double>>& OutputW, vector<double>& Bias, vector<vector<vector<double>>>& chromosomeWeights )
{
    vector<pair<double, int>> vp;
    vector<int> chosenNeurons(hiddenNeuronsN, 0);

    //сортировка по приспособленности
    for (int i = 0; i < subpopulationsN; i++)
    {
        vp.clear();
        for (int j = 0; j < speciesN; j++)
        {
            vp.push_back(make_pair(chromosomeNormNeuronFitness[i][j], j));
        }
        sort(vp.rbegin(), vp.rend());

        chosenNeurons[i] = vp[0].second;
        int j = chosenNeurons[i];
        // инициализация сети
          // neuron inputs
        for (int k = 0; k < inputN; k++)
        {
            InputW[i][k] = chromosomeWeights[i][j][k];
        }
        // neuron outputs
        for (int k = inputN; k < inputN + outputN; k++)
        {
            OutputW[i][k - inputN] = chromosomeWeights[i][j][k];
        }
        // neuron bias
        Bias[i] = chromosomeWeights[i][j][gene_lenght - 1];
    }
}

int main()
{
    // вектор со значениями датасета
    vector<vector<double>> dataset(dataset_rows);

    // считывание данных датасета - start
    ifstream ifile("cancer1.dt", ios::in);

    //check to see that the file was opened correctly:
    if (!ifile.is_open()) {
        cerr << "There was a problem opening the input file!\n";
        exit(1);//exit or do additional error checking
    }

    double num = 0.0;
    int i = 0, j = 0;
    //keep storing values from the text file so long as data exists:
    while (ifile >> num) {
        dataset[i].push_back(num);
        j++;
        if (j == 11)
        {
            i++;
            j = 0;
        }
    }
    ifile.close();
    // считывание данных датасета - end

    // вектора с данными для обучения и тестирования
    std::vector<vector<double>> training_set(int(dataset_rows * coef + 1));
    std::vector<vector<double>> testing_set(int(dataset_rows * (1 - coef) + 1));

    for (int i = 0; i < dataset_rows * coef; i++)
    {
        training_set[i] = vector<double>(col);
        for (int j = 0; j < col; j++)
        {
            training_set[i][j] = dataset[i][j];
        }
    }

    for (int i = (int)dataset_rows * coef; i < dataset_rows; i++)
    {
        testing_set[i - (int)dataset_rows * coef] = vector<double>(col);
        for (int j = 0; j < col; j++)
        {
            testing_set[i - (int)dataset_rows * coef][j] = dataset[i][j];
        }
    }

    srand(static_cast<unsigned int>(time(0)));

    //Chromosomes
    // массив для хранения генов хромосом сабпопуляций (вх + вых + bias)
    vector<vector<vector<double>>> chromosomeWeights(subpopulationsN, vector<vector<double>>(speciesN, vector<double>(gene_lenght, 0)));
    // счётчик задействований нейронов в trials
    vector<vector<int>> chromosomeCount(subpopulationsN, vector<int>(speciesN, 0));
    // приспособленность нейронов
    vector<vector<double>> chromosomeNeuronFitness(subpopulationsN, vector<double>(speciesN, 0));
    // нормализованная приспособленность нейронов
    vector<vector<double>> chromosomeNormNeuronFitness(subpopulationsN, vector<double>(speciesN, 0));

    //инициализация хромосом
    //подпопуляции
    for (int i = 0; i < subpopulationsN; i++)
    {
        //особи (нейроны)
        for (int j = 0; j < speciesN; j++)
        {
            for (int k = 0; k < gene_lenght; k++)
            {
                chromosomeWeights[i][j][k] = ((double)rand() / (RAND_MAX)) - 0.5;
            }
        }
    }

    // Neural Net ( trials)
    vector<double>TRneuralNetInput(inputN, 0);                                                  //входы сети
    vector<vector<double>> TRneuralNetHiddenInputW(hiddenNeuronsN, vector<double>(inputN, 0));  //входные веса нейронов скрытого слоя
    vector<vector<double>> TRneuralNetHiddenOutputW(hiddenNeuronsN, vector<double>(outputN, 0));//выходные веса нейронов скрытого слоя
    vector<double>TRneuralNetHiddenBias (hiddenNeuronsN, 0);                                    //смещения нейронов скрытого слоя

//ESP - algorithm
    double best_fitness = 0;
    double goal_fitness = 0.95;
    double mean_fitness = 0;
    double current_fitness = 0;
    int noImprovements = 0;
    vector<int> chosenNeurons(subpopulationsN);
    //буфер для сортировки 
    vector<pair<double, int>> vp;

    // счётчик итераций
    int counter = 0;
    // макс. итерация
    int max_counter = 100;
    // счётчик повторений стагнации
    int consequent_stagnation = 0;
    // средняя приспосабливаемость за каждый trial
    vector<double> mean_fitness_trials;
    mean_fitness_trials.push_back(0);

    //best_fitness < goal_fitness &&
    while ( mean_fitness < goal_fitness)
    {
    // Evaluation
        // Trials 
        for (int trial = 1; trial < hiddenNeuronsN * 10; trial++)
        {
            //создаём случайную сеть - start
                //Проходим по подпопуляциям/нейронам в создаваемой сети
            for (int i = 0; i < subpopulationsN; i++)
            {
                //cout << "Subpopulation: " << i << endl;
                int j = rand() % speciesN;
                chosenNeurons[i] = j;
                chromosomeCount[i][j] += 1;
                // neuron inputs
                for (int k = 0; k < inputN; k++)
                {
                    TRneuralNetHiddenInputW[i][k] = chromosomeWeights[i][j][k];
                }
                // neuron outputs
                for (int k = inputN; k < inputN + outputN; k++)
                {
                    TRneuralNetHiddenOutputW[i][k - inputN] = chromosomeWeights[i][j][k];
                }
                // neuron bias
                TRneuralNetHiddenBias[i] = chromosomeWeights[i][j][gene_lenght - 1];
            }
            //создаём случайную сеть - end

            // подготовка и пропуск данных через нейронную сеть
            bool CorrectOutput = PrepareTrainingData(training_set, TRneuralNetInput);
            current_fitness = EvaluateNet(TRneuralNetInput, TRneuralNetHiddenInputW, TRneuralNetHiddenOutputW, TRneuralNetHiddenBias, CorrectOutput);

            // подсчёт приспосабливаемости
            for (int i = 0; i < subpopulationsN; i++)
            {
                chromosomeNeuronFitness[i][chosenNeurons[i]] += current_fitness;
            }

            // сохраняем лучшее значение приспосабливаемости
            if (current_fitness > mean_fitness_trials.back())
            {
                best_fitness = current_fitness;
                noImprovements = 0;
            }
            else
            {
                noImprovements++;
            }

           // cout << counter << ": " << current_fitness << endl;
            mean_fitness += current_fitness;
        }
        // The end of Trials

        //нормализация приспособленности - start
        for (int i = 0; i < subpopulationsN; i++)
        {
            for (int j = 0; j < speciesN; j++)
            {
                if (chromosomeCount[i][j] == 0) { chromosomeCount[i][j] = 1; }
                chromosomeNormNeuronFitness[i][j] = chromosomeNeuronFitness[i][j] / (double)chromosomeCount[i][j];
                chromosomeCount[i][j] = 0;
                chromosomeNeuronFitness[i][j] = 0;
            }
        }
        // нормализация приспособленности - end

        mean_fitness = mean_fitness / (hiddenNeuronsN * 10);
        mean_fitness_trials.push_back(mean_fitness);
        cout << counter << " mean: " << mean_fitness << endl;
        counter++;

//Check Stagnation----------------------------------------------------------------------------------------------------------------------------------------
        // проверка на стагнацию
        if (noImprovements > 5)
        {
            consequent_stagnation++;
            if (consequent_stagnation == 2)
            {
                consequent_stagnation = 0;
                bool Neuron_removed = false;
                bool CorrectOutput;
                double initial_fitness;
                double lesioned_fitness;

                GetBestNetwork(chromosomeNormNeuronFitness, TRneuralNetHiddenInputW, TRneuralNetHiddenOutputW, TRneuralNetHiddenBias, chromosomeWeights);

                do {
                    CorrectOutput = PrepareTrainingData(training_set, TRneuralNetInput);
                    initial_fitness = EvaluateNet(TRneuralNetInput, TRneuralNetHiddenInputW, TRneuralNetHiddenOutputW, TRneuralNetHiddenBias, CorrectOutput);
                } while (initial_fitness < 0.7); // пока выполняется условие.

                for (int i = 0; i < hiddenNeuronsN; i++)
                {
                    CorrectOutput = PrepareTrainingData(testing_set, TRneuralNetInput);
                    lesioned_fitness = EvaluateLesionedNet(TRneuralNetInput, TRneuralNetHiddenInputW, TRneuralNetHiddenOutputW, TRneuralNetHiddenBias, CorrectOutput, i);
                    
                    if (lesioned_fitness > initial_fitness)
                    {
                        Neuron_removed = true;

                        subpopulationsN += -1;
                        hiddenNeuronsN += -1;

                        chosenNeurons.erase(chosenNeurons.begin() + i);

                        TRneuralNetHiddenInputW.erase(TRneuralNetHiddenInputW.begin() + i);
                        TRneuralNetHiddenOutputW.erase(TRneuralNetHiddenOutputW.begin() + i);
                        TRneuralNetHiddenBias.erase(TRneuralNetHiddenBias.begin() + i);

                        chromosomeWeights.erase(chromosomeWeights.begin() + i);
                        chromosomeCount.erase(chromosomeCount.begin() + i);
                        chromosomeNeuronFitness.erase(chromosomeNeuronFitness.begin() + i);
                        chromosomeNormNeuronFitness.erase(chromosomeNormNeuronFitness.begin() + i);
                        
                    }
                }

                if (!Neuron_removed)
                {
                    subpopulationsN += 1;
                    hiddenNeuronsN += 1;

                    chosenNeurons.push_back(0);

                    vector<vector<double>> newRowWeights(speciesN, vector<double>(gene_lenght, 0));
                    chromosomeWeights.push_back(newRowWeights);

                    vector<double> newRowSpecies_double(speciesN, 0);
                    chromosomeNeuronFitness.push_back(newRowSpecies_double);
                    chromosomeNormNeuronFitness.push_back(newRowSpecies_double);

                    vector<int> newRowSpecies_int(speciesN, 0);
                    chromosomeCount.push_back(newRowSpecies_int);

                    vector<double> newRowInput(inputN, 0);
                    vector<double> newRowOutput(inputN, 0);

                    TRneuralNetHiddenInputW.push_back(newRowInput);
                    TRneuralNetHiddenOutputW.push_back(newRowOutput);
                    TRneuralNetHiddenBias.push_back(0);
                }
            }
            else
            {
                BurstMutate(chromosomeWeights, chromosomeNormNeuronFitness);
            }
            
        }

    // Рекомбинация---------------------------------------------------------------------------------------------------------------------------------------------
        // Neuron Level - start
        //сортировка нейронов подпопуляций по нормализованной приспособленности
        for (int i = 0; i < subpopulationsN; i++)
        {
            vp.clear();
            // сохраняем нормализованный fitness с индексом после сортировки
            for (int j = 0; j < speciesN; j++)
            {
                vp.push_back(make_pair(chromosomeNormNeuronFitness[i][j], j));
            }
            sort(vp.rbegin(), vp.rend());

            // скрещивание
            for (int j = 0; j < speciesN / 4; j++)
            {
                int firstNeuron = vp[0].second;
                int secondNeuron = vp[j+1].second;
                int fdNeuron = vp[(j + 1) * 2 + 1].second;
                int sdNeuron = vp[(j + 1) * 2 + 2].second;

                crossoverNeuronLevel(chromosomeWeights, i, firstNeuron, secondNeuron, fdNeuron, sdNeuron);
            }
            // мутация
            for (int j = (speciesN / 2) + 2; j < speciesN; j++)
            {
                mutate(chromosomeWeights, i, vp[j].second);
            }
        }
        // Neuron Level - end
    }
    cout << "End of training!" << endl;

    //создание сети из лучших нейронов
    GetBestNetwork(chromosomeNormNeuronFitness, TRneuralNetHiddenInputW, TRneuralNetHiddenOutputW, TRneuralNetHiddenBias, chromosomeWeights);

    double accuracy = 0;
    // тестирование сети
    for (int i = 0; i < testing_set.size(); i++)
    {
        bool CorrectOutput = PrepareTrainingData(testing_set, TRneuralNetInput);
        current_fitness = EvaluateNet(TRneuralNetInput, TRneuralNetHiddenInputW, TRneuralNetHiddenOutputW, TRneuralNetHiddenBias, CorrectOutput);
        accuracy += current_fitness;
        cout << "Test " << i << ": " << current_fitness << endl;
    }
    accuracy /= testing_set.size();
    cout << "accuracy: " << accuracy << endl;

    // сохранение связей в файл 
    ofstream outfile;
    outfile.open("weights.txt", std::ofstream::out | std::ofstream::trunc);

    for (int i = 0; i < hiddenNeuronsN; i++)
    {
        for (int k = 0; k < inputN; k++)
        {
            outfile << TRneuralNetHiddenInputW[i][k] << " ";
        }
        // neuron outputs
        for (int k = inputN; k < inputN + outputN; k++)
        {
            outfile << TRneuralNetHiddenOutputW[i][k - inputN] << " ";
        }
        // neuron bias
        outfile << TRneuralNetHiddenBias[i] << endl;
    }
    outfile.close();

    // сохранение динамики приспосабливаемости в файл 
    outfile.open("fitness.txt", std::ofstream::out | std::ofstream::trunc);

    for (int i = 0; i < mean_fitness_trials.size(); i++)
    {
        outfile << mean_fitness_trials[i] << endl;
    }
    outfile.close();
}