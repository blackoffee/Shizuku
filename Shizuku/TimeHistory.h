#pragma once

#include <queue>
#include "Shizuku.Flow/TimerKey.h"
#include "Shizuku.Core/Types/MinMax.h"

#define DEFAULT_HISTORY_LENGTH 128

using namespace Shizuku::Flow;

namespace Shizuku{ namespace Presentation{
    class TimeHistory
    {
    private:
        int m_size;
        std::deque<double> m_queue;
        Shizuku::Core::Types::MinMax<double> m_minMax;

    public:
        TimeHistory();
        TimeHistory(const int p_size);
        void Append(const double p_value);
        float DataProvider(void* p_data, int p_index);
        void Resize(const int p_size);
        int Size();
        Shizuku::Core::Types::MinMax<double> MinMax();

        static TimeHistory& Instance(const TimerKey p_key)
        {
            static TimeHistory s_solve = TimeHistory(DEFAULT_HISTORY_LENGTH);
            static TimeHistory s_prepareSurface = TimeHistory(DEFAULT_HISTORY_LENGTH);
            static TimeHistory s_prepareFloor = TimeHistory(DEFAULT_HISTORY_LENGTH);
            static TimeHistory s_processSurface = TimeHistory(DEFAULT_HISTORY_LENGTH);
            static TimeHistory s_processFloor = TimeHistory(DEFAULT_HISTORY_LENGTH);
            switch (p_key)
            {
            case TimerKey::SolveFluid:
                return s_solve;
            case TimerKey::PrepareSurface:
                return s_prepareSurface;
            case TimerKey::PrepareFloor:
                return s_prepareFloor;
            case TimerKey::ProcessSurface:
                return s_processSurface;
            case TimerKey::ProcessFloor:
                return s_processFloor;
            }

            throw "Unknown TimerKey";
        }

        static void SetHistoryLength(const int p_length)
        {
            Instance(TimerKey::SolveFluid).Resize(p_length);
            Instance(TimerKey::PrepareFloor).Resize(p_length);
            Instance(TimerKey::PrepareSurface).Resize(p_length);
            Instance(TimerKey::ProcessFloor).Resize(p_length);
            Instance(TimerKey::ProcessSurface).Resize(p_length);
        }
    };
} }