#pragma once

#include <boost/any.hpp>

#ifdef SHIZUKU_FLOW_EXPORTS  
#define FLOW_API __declspec(dllexport)   
#else  
#define FLOW_API __declspec(dllimport)   
#endif  

namespace Shizuku{ namespace Flow{
    class Flow;
} }

namespace Shizuku{ namespace Flow{ namespace Command{
    class FLOW_API Command
    {
    protected:
        Flow* m_flow;
        enum State { Active, Inactive };
        State m_state;
        Command();
        Command(Flow& p_flow);
        void Start();
        void Start(boost::any const p_param);
        void Track();
        void Track(boost::any const p_param);
        void End();
        void End(boost::any const p_param);
    };
} } }
