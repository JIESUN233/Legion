// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2020, Intel Corporation
#pragma once
//written by Roman Sudarikov

#include <iostream>
#include "cpucounters.h"
#include "utils.h"
#include <vector>
#include <array>
#include <string>
#include <initializer_list>
#include <algorithm>

#if defined(_MSC_VER)
typedef unsigned int uint;
#endif

using namespace std;
using namespace pcm;
#define NUM_SAMPLES (1)

static void print(const vector<string> &listNames, bool csv)
{
    for(auto& name : listNames)
        if (csv)
            cout << "," << name;
        else
            cout << "|  " << name << "  ";
}

static uint getIdent (const string &s)
{
    /*
     * We are adding "|  " before and "  " after the event name hence +5 to
     * strlen(eventNames). Rest of the logic is to center the event name.
     */
    uint ident = 5 + (uint)s.size();
    return (3 + ident / 2);
}

class IPlatform
{
    void init();

public:
    IPlatform(PCM *m, bool csv, bool bandwidth, bool verbose);
    virtual void getEvents(int flag) = 0;
    virtual void getEventGroup(eventGroup_t &eventGroup, int flag) = 0;
    virtual vector<eventGroup_t> *getEventGroups() = 0;

    virtual void printHeader() = 0;
    virtual void printEvents() = 0;
    virtual void printAggregatedEvents() = 0;
    virtual void cleanup() = 0;

    virtual std::vector<uint64_t> GetCounter() = 0;
    // virtual PCM *getPCM() = 0;

    static IPlatform *getPlatform(PCM* m, bool csv, bool bandwidth,
                                        bool verbose, uint32 delay);
    virtual ~IPlatform() { }

protected:
    PCM *m_pcm;
    bool m_csv;
    bool m_bandwidth;
    bool m_verbose;
    uint m_socketCount;

    enum eventFilter {TOTAL, MISS, HIT, fltLast};

    vector<string> filterNames, bwNames;
};

void IPlatform::init()
{
    print_cpu_details();

    if (m_pcm->isSomeCoreOfflined())
    {
        cerr << "Core offlining is not supported. Program aborted\n";
        exit(EXIT_FAILURE);
    }
}

IPlatform::IPlatform(PCM *m, bool csv, bool bandwidth, bool verbose) :
        // m_pcm(m),
        filterNames {"(Total)", "(Miss)", "(Hit)"},
        bwNames {"PCIe Rd (B)", "PCIe Wr (B)"}
{
    m_pcm = m;
    m_csv = csv;
    m_bandwidth = bandwidth;
    m_verbose = verbose;
    m_socketCount = m_pcm->getNumSockets();

    init();
}

/*
 * Common API to program, access and represent required Uncore counters.
 * The only difference is event opcodes and the way how bandwidth is calculated.
 */
class LegacyPlatform: public IPlatform
{
    enum {
        before,
        after,
        total
    };
    vector<string> eventNames;
    
    uint32 m_delay;
    typedef vector <vector <uint64>> eventCount_t;
    array<eventCount_t, total> eventCount;
    vector<eventGroup_t> eventGroups;
    virtual void getEvents(int flag) final;
    virtual void getEventGroup(eventGroup_t &eventGroup, int flag) final;
    // virtual PCM *getPCM() final;

    virtual void printHeader() final;
    virtual void printEvents() final;
    virtual void printAggregatedEvents() final;
    virtual void cleanup() final;
    // virtual vector<eventGroup_t> *getEventGroups() final;
    
    void printBandwidth(uint socket, eventFilter filter);
    void printBandwidth();
    void printSocketScopeEvent(uint socket, eventFilter filter, uint idx);
    void printSocketScopeEvents(uint socket, eventFilter filter);
    uint64 getEventCount (uint socket, uint idx);
    uint eventGroupOffset(eventGroup_t &eventGroup);
    
    void printAggregatedEvent(uint idx);

    std::vector<uint64_t> GetCounter();

public:
    
    LegacyPlatform(initializer_list<string> events, initializer_list <eventGroup_t> eventCodes,
        PCM *m, bool csv, bool bandwidth, bool verbose, uint32 delay) :
        IPlatform(m, csv, bandwidth, verbose),
            eventNames(events), eventGroups(eventCodes)
    {
        int eventsCount = 0;
        for (auto &group : eventGroups) eventsCount += (int)group.size();

        m_delay = uint32(delay * 1000 / (eventGroups.size()) / NUM_SAMPLES);
        if (m_delay * eventsCount * NUM_SAMPLES < delay * 1000) ++m_delay;

        eventSample.resize(m_socketCount);
        for (auto &e: eventSample)
            e.resize(eventsCount);

        for (auto &run : eventCount) {
            run.resize(m_socketCount);
            for (auto &events_ : run)
                events_.resize(eventsCount);
        }
    };

    vector<eventGroup_t> *getEventGroups(){
        return &eventGroups;
    }

protected:
    vector<vector<uint64>> eventSample;
    virtual uint64 getReadBw(uint socket, eventFilter filter) = 0;
    virtual uint64 getWriteBw(uint socket, eventFilter filter) = 0;
    virtual uint64 getReadBw() = 0;
    virtual uint64 getWriteBw() = 0;
    virtual uint64 event(uint socket, eventFilter filter, uint idx) = 0;
};

std::vector<uint64_t> LegacyPlatform::GetCounter()
{
    std::vector<uint64_t> ret;
    for(uint skt =0; skt < 2; ++skt){
        uint64 value = event(skt, TOTAL, 0);
        ret.push_back(value);
        cout<<skt<< " value " << value<<"\n";
    }
    return ret;
}


void LegacyPlatform::cleanup()
{
    for(auto& socket : eventSample)
        fill(socket.begin(), socket.end(), 0);
}

inline uint64 LegacyPlatform::getEventCount (uint skt, uint idx)
{
    return eventGroups.size() * (eventCount[after][skt][idx] - eventCount[before][skt][idx]);
}

uint LegacyPlatform::eventGroupOffset(eventGroup_t &eventGroup)
{
    uint offset = 0;
    uint grpIdx = (uint)(&eventGroup - eventGroups.data());

    for (auto iter = eventGroups.begin(); iter < eventGroups.begin() + grpIdx; iter++)
         offset += (uint)iter->size();

    return offset;
}



void LegacyPlatform::getEventGroup(eventGroup_t &eventGroup, int flag)
{
    // 指定对应监控事件
    // m_pcm->programPCIeEventGroup(eventGroup);
    uint offset = eventGroupOffset(eventGroup);
    // cout<<"offset:"<<offset<<endl;

    if(flag == 0){
        for(uint skt =0; skt < m_socketCount; ++skt)
            for (uint ctr = 0; ctr < eventGroup.size(); ++ctr){
                eventCount[0][skt][ctr + offset] = m_pcm->getPCIeCounterData(skt, ctr);
                // cout << skt<<" "<< getEventCountBefore(skt, ctr + offset)<<endl;
                // cout<<"socket-"<<skt<<":"<<eventCount[0][skt][ctr + offset]<<endl;
            }
        
    }else{
        // MySleepMs(uint(20));    
        for(uint skt =0; skt < m_socketCount; ++skt)
            for (uint ctr = 0; ctr < eventGroup.size(); ++ctr){
                eventCount[1][skt][ctr + offset] = m_pcm->getPCIeCounterData(skt, ctr);
                // cout << skt<<" "<< getEventCountAfter(skt, ctr + offset)<<endl;
                // cout<<"socket-"<<skt<<":"<<eventCount[1][skt][ctr + offset]<<endl;
            }
                
        for(uint skt = 0; skt < m_socketCount; ++skt)
            for (uint idx = offset; idx < offset + eventGroup.size(); ++idx){
                // if(skt == 0 && idx == 0){
                    // cout<<"before:"<<eventSample[skt][idx]<<endl;
                    eventSample[skt][idx] += getEventCount(skt, idx);
                    // cout<<"after:"<<eventSample[skt][idx]<<endl;
            }   
    }

    // 1-监控循环
    // for (auto &run : eventCount) {
    //     // cout<<x++<<endl;
    //     for(uint skt =0; skt < m_socketCount; ++skt)
    //         for (uint ctr = 0; ctr < eventGroup.size(); ++ctr)
    //             run[skt][ctr + offset] = m_pcm->getPCIeCounterData(skt, ctr);

    //     MySleepMs(uint(m_delay));
    // }

    // 2-获取监控数据
    // for(uint skt = 0; skt < m_socketCount; ++skt)
    //     for (uint idx = offset; idx < offset + eventGroup.size(); ++idx){
    //         eventSample[skt][idx] += getEventCount(skt, idx);
    //     }
}


void LegacyPlatform::getEvents(int flag)
{
    cout<<&eventGroups<<endl;

    for (auto& evGroup : eventGroups){
        // cout<<evGroup.size()<<endl;
        for (auto x : evGroup){
            cout<<&x<<endl;
            // cout<<"opcodes: "<< x <<endl;
        }
        // m_pcm->programPCIeEventGroup(evGroup);
        // if(flag)
        //     MySleepMs(uint(10));
        getEventGroup(evGroup, flag);
    }
}


void LegacyPlatform::printHeader()
{
    cout << "Skt";
    if (!m_csv)
        cout << ' ';

    print(eventNames, m_csv);
    if (m_bandwidth)
        print(bwNames, m_csv);

    cout << "\n";
}

void LegacyPlatform::printBandwidth(uint skt, eventFilter filter)
{
    typedef uint64 (LegacyPlatform::*bwFunc_t)(uint, eventFilter);
    vector<bwFunc_t> bwFunc = {
        &LegacyPlatform::getReadBw,
        &LegacyPlatform::getWriteBw,
    };

    if (!m_csv)
        for(auto& bw_f : bwFunc) {
            int ident = getIdent(bwNames[&bw_f - bwFunc.data()]);
            cout << setw(ident)
                 << unit_format((this->*bw_f)(skt,filter))
                //  << (this->*bw_f)(skt,filter)
                 << setw(5 + bwNames[&bw_f - bwFunc.data()].size() - ident)
                 << ' ';
        }
    else
        for(auto& bw_f : bwFunc)
            cout << ',' << (this->*bw_f)(skt,filter);
}

void LegacyPlatform::printSocketScopeEvent(uint skt, eventFilter filter, uint idx)
{
    uint64 value = event(skt, filter, idx);

    if (m_csv)
        cout << ',' << value;
    else
    {
        int ident = getIdent(eventNames[idx]);
        cout << setw(ident)
             << unit_format(value)
            //  << value
             << setw(5 + eventNames[idx].size() - ident)
             << ' ';
    }
}

void LegacyPlatform::printSocketScopeEvents(uint skt, eventFilter filter)
{
    if (!m_csv) {
        int ident = (int)strlen("Skt |") / 2;
        cout << setw(ident) << skt << setw(ident) << ' ';
    } else
        cout << skt;

    for(uint idx = 0; idx < eventNames.size(); ++idx)
        printSocketScopeEvent(skt, filter, idx);

    if (m_bandwidth)
        printBandwidth(skt, filter);

    if(m_verbose)
        cout << filterNames[filter];

    cout << "\n";
}

void LegacyPlatform::printEvents()
{
    for(uint skt =0; skt < m_socketCount; ++skt)
        if (!m_verbose)
            printSocketScopeEvents(skt, TOTAL);
        else
            for (uint flt = TOTAL; flt < fltLast; ++flt)
                printSocketScopeEvents(skt, static_cast<eventFilter>(flt));
}

void LegacyPlatform::printAggregatedEvent(uint idx)
{
    uint64 value = 0;
    for(uint skt =0; skt < m_socketCount; ++skt)
        value += event(skt, TOTAL, idx);

    int ident = getIdent(eventNames[idx]);
    cout << setw(ident)
         << unit_format(value)
         << setw(5 + eventNames[idx].size() - ident) << ' ';
}

void LegacyPlatform::printBandwidth()
{
    typedef uint64 (LegacyPlatform::*bwFunc_t)();
    vector<bwFunc_t> bwFunc = {
        &LegacyPlatform::getReadBw,
        &LegacyPlatform::getWriteBw,
    };

    for(auto& bw_f : bwFunc) {
        int ident = getIdent(bwNames[&bw_f - bwFunc.data()]);
        cout << setw(ident)
             << unit_format((this->*bw_f)())
             << setw(5 + bwNames[&bw_f - bwFunc.data()].size() - ident)
             << ' ';
    }
}

void LegacyPlatform::printAggregatedEvents()
{
    if (!m_csv)
    {
        uint len = (uint)strlen("Skt ");

        for(auto& evt : eventNames)
            len += (5 + (uint)evt.size());

        if (m_bandwidth)
            for(auto& bw : bwNames)
                len += (5 + (uint)bw.size());

        while (len--)
            cout << '-';
        cout << "\n";

        int ident = (int)strlen("Skt |") /2 ;
        cout << setw(ident) << "*" << setw(ident) << ' ';

        for (uint idx = 0; idx < eventNames.size(); ++idx)
            printAggregatedEvent(idx);

        if (m_bandwidth)
            printBandwidth();

        if (m_verbose)
            cout << "(Aggregate)\n\n";
        else
            cout << "\n\n";
    }
}

//SPR
class EagleStreamPlatform: public LegacyPlatform
{
public:
    EagleStreamPlatform(PCM *m, bool csv, bool bandwidth, bool verbose, uint32 delay) :
        LegacyPlatform( {"PCIRdCur", "ItoM", "ItoMCacheNear", "UCRdF", "WiL", "WCiL", "WCiLF"},
                        {
                            {0xC8F3FE00000435, 0xC8F3FD00000435, 0xCC43FE00000435, 0xCC43FD00000435},
                            {0xCD43FE00000435, 0xCD43FD00000435, 0xC877DE00000135, 0xC87FDE00000135},
                            {0xC86FFE00000135, 0xC867FE00000135,},
                        },
                        m, csv, bandwidth, verbose, delay)
    {
    };

private:
    enum eventIdx {
        PCIRdCur,
        ItoM,
        ItoMCacheNear,
        UCRdF,
        WiL,
        WCiL,
        WCiLF
    };

    enum Events {
            PCIRdCur_miss,
            PCIRdCur_hit,
            ItoM_miss,
            ItoM_hit,
            ItoMCacheNear_miss,
            ItoMCacheNear_hit,
            UCRdF_miss,
            WiL_miss,
            WCiL_miss,
            WCiLF_miss,
            eventLast
    };

    virtual uint64 getReadBw(uint socket, eventFilter filter);
    virtual uint64 getWriteBw(uint socket, eventFilter filter);
    virtual uint64 getReadBw();
    virtual uint64 getWriteBw();
    virtual uint64 event(uint socket, eventFilter filter, uint idx);
};

uint64 EagleStreamPlatform::event(uint socket, eventFilter filter, uint idx)
{
    uint64 event = 0;
    switch (idx)
    {
        case PCIRdCur:
            if(filter == TOTAL)
                event = eventSample[socket][PCIRdCur_miss] +
                        eventSample[socket][PCIRdCur_hit];
                else if (filter == MISS)
                    event = eventSample[socket][PCIRdCur_miss];
                else if (filter == HIT)
                    event = eventSample[socket][PCIRdCur_hit];
            break;
        case ItoM:
            if(filter == TOTAL)
                event = eventSample[socket][ItoM_miss] +
                        eventSample[socket][ItoM_hit];
                else if (filter == MISS)
                    event = eventSample[socket][ItoM_miss];
                else if (filter == HIT)
                    event = eventSample[socket][ItoM_hit];
            break;
        case ItoMCacheNear:
            if(filter == TOTAL)
                event = eventSample[socket][ItoMCacheNear_miss] +
                        eventSample[socket][ItoMCacheNear_hit];
                else if (filter == MISS)
                    event = eventSample[socket][ItoMCacheNear_miss];
                else if (filter == HIT)
                    event = eventSample[socket][ItoMCacheNear_hit];
            break;
        case UCRdF:
                if(filter == TOTAL || filter == MISS)
                    event = eventSample[socket][UCRdF_miss];
            break;
        case WiL:
                if(filter == TOTAL || filter == MISS)
                    event = eventSample[socket][WiL_miss];
            break;
        case WCiL:
                if(filter == TOTAL || filter == MISS)
                    event = eventSample[socket][WCiL_miss];
            break;
        case WCiLF:
                if(filter == TOTAL || filter == MISS)
                    event = eventSample[socket][WCiLF_miss];
            break;
        default:
            break;
    }
    return event;
}

uint64 EagleStreamPlatform::getReadBw(uint socket, eventFilter filter)
{
    uint64 readBw = event(socket, filter, PCIRdCur);
    return (readBw * 64ULL);
}

uint64 EagleStreamPlatform::getWriteBw(uint socket, eventFilter filter)
{
    uint64 writeBw = event(socket, filter, ItoM) +
                     event(socket, filter, ItoMCacheNear);
    return (writeBw * 64ULL);
}
uint64 EagleStreamPlatform::getReadBw()
{
    uint64 readBw = 0;
    for (uint socket = 0; socket < m_socketCount; socket++)
        readBw += (event(socket, TOTAL, PCIRdCur));
    return (readBw * 64ULL);
}

uint64 EagleStreamPlatform::getWriteBw()
{
    uint64 writeBw = 0;
    for (uint socket = 0; socket < m_socketCount; socket++)
        writeBw += (event(socket, TOTAL, ItoM) +
                    event(socket, TOTAL, ItoMCacheNear));
    return (writeBw * 64ULL);
}

//ICX
class WhitleyPlatform: public LegacyPlatform
{
public:
    WhitleyPlatform(PCM *m, bool csv, bool bandwidth, bool verbose, uint32 delay) :
        LegacyPlatform( {"PCIRdCur", "ItoM", "ItoMCacheNear", "UCRdF","WiL"},
                        {
                            {0xC8F3FE00000435},
                            // {0xC8F3FE00000435, 0xC8F3FD00000435, 0xCC43FE00000435, 0xCC43FD00000435},
                            // {0xCD43FE00000435, 0xCD43FD00000435, 0xC877DE00000135, 0xC87FDE00000135},
                        },
                        m, csv, bandwidth, verbose, delay)
    {
    };

private:
    enum eventIdx {
        PCIRdCur,
        ItoM,
        ItoMCacheNear,
        UCRdF,
        WiL,
    };

    enum Events {
            PCIRdCur_miss,
            PCIRdCur_hit,
            ItoM_miss,
            ItoM_hit,
            ItoMCacheNear_miss,
            ItoMCacheNear_hit,
            UCRdF_miss,
            WiL_miss,
            eventLast
    };

    virtual uint64 getReadBw(uint socket, eventFilter filter);
    virtual uint64 getWriteBw(uint socket, eventFilter filter);
    virtual uint64 getReadBw();
    virtual uint64 getWriteBw();
    virtual uint64 event(uint socket, eventFilter filter, uint idx);
};

uint64 WhitleyPlatform::event(uint socket, eventFilter filter, uint idx)
{
    uint64 event = 0;
    switch (idx)
    {
        case PCIRdCur:
            if(filter == TOTAL)
                event = eventSample[socket][PCIRdCur_miss] +
                        eventSample[socket][PCIRdCur_hit];
                else if (filter == MISS)
                    event = eventSample[socket][PCIRdCur_miss];
                else if (filter == HIT)
                    event = eventSample[socket][PCIRdCur_hit];
            break;
        case ItoM:
            if(filter == TOTAL)
                event = eventSample[socket][ItoM_miss] +
                        eventSample[socket][ItoM_hit];
                else if (filter == MISS)
                    event = eventSample[socket][ItoM_miss];
                else if (filter == HIT)
                    event = eventSample[socket][ItoM_hit];
            break;
        case ItoMCacheNear:
            if(filter == TOTAL)
                event = eventSample[socket][ItoMCacheNear_miss] +
                        eventSample[socket][ItoMCacheNear_hit];
                else if (filter == MISS)
                    event = eventSample[socket][ItoMCacheNear_miss];
                else if (filter == HIT)
                    event = eventSample[socket][ItoMCacheNear_hit];
            break;
        case UCRdF:
                if(filter == TOTAL || filter == MISS)
                    event = eventSample[socket][UCRdF_miss];
            break;
        case WiL:
                if(filter == TOTAL || filter == MISS)
                    event = eventSample[socket][WiL_miss];
            break;
        default:
            break;
    }
    return event;
}

uint64 WhitleyPlatform::getReadBw(uint socket, eventFilter filter)
{
    uint64 readBw = event(socket, filter, PCIRdCur);
    return (readBw * 64ULL);
}

uint64 WhitleyPlatform::getWriteBw(uint socket, eventFilter filter)
{
    uint64 writeBw = event(socket, filter, ItoM) +
                     event(socket, filter, ItoMCacheNear);
    return (writeBw * 64ULL);
}
uint64 WhitleyPlatform::getReadBw()
{
    uint64 readBw = 0;
    for (uint socket = 0; socket < m_socketCount; socket++)
        readBw += (event(socket, TOTAL, PCIRdCur));
    return (readBw * 64ULL);
}

uint64 WhitleyPlatform::getWriteBw()
{
    uint64 writeBw = 0;
    for (uint socket = 0; socket < m_socketCount; socket++)
        writeBw += (event(socket, TOTAL, ItoM) +
                    event(socket, TOTAL, ItoMCacheNear));
    return (writeBw * 64ULL);
}

// CLX, SKX
class PurleyPlatform: public LegacyPlatform
{
public:
    PurleyPlatform(PCM *m, bool csv, bool bandwidth, bool verbose, uint32 delay) :
        LegacyPlatform( {"PCIRdCur", "RFO", "CRd", "DRd","ItoM", "PRd", "WiL"},
                        {
                            {0x00043c33}, //PCIRdCur_miss
                            // {0x00043c37}, //PCIRdCur_hit
                            // {0x00040033}, //RFO_miss
                            // {0x00040037}, //RFO_hit
                            // {0x00040233}, //CRd_miss
                            // {0x00040237}, //CRd_hit
                            // {0x00040433}, //DRd_miss
                            // {0x00040437}, //DRd_hit
                            // {0x00049033}, //ItoM_miss
                            // {0x00049037}, //ItoM_hit
                            // {0x40040e33}, //PRd_miss
                            // {0x40040e37}, //PRd_hit
                            // {0x40041e33}, //WiL_miss
                            // {0x40041e37}, //WiL_hit
                        },
                        m, csv, bandwidth, verbose, delay)
    {
    };

private:
    enum eventIdx {
        PCIRdCur,
        RFO,
        CRd,
        DRd,
        ItoM,
        PRd,
        WiL,
    };

    enum Events {
            PCIRdCur_miss,
            PCIRdCur_hit,
            RFO_miss,
            RFO_hit,
            CRd_miss,
            CRd_hit,
            DRd_miss,
            DRd_hit,
            ItoM_miss,
            ItoM_hit,
            PRd_miss,
            PRd_hit,
            WiL_miss,
            WiL_hit,
    };

    virtual uint64 getReadBw(uint socket, eventFilter filter);
    virtual uint64 getWriteBw(uint socket, eventFilter filter);
    virtual uint64 getReadBw();
    virtual uint64 getWriteBw();
    virtual uint64 event(uint socket, eventFilter filter, uint idx);
};

uint64 PurleyPlatform::event(uint socket, eventFilter filter, uint idx)
{
    uint64 event = 0;
    if(filter == TOTAL)
        event = eventSample[socket][2 * idx] +
                eventSample[socket][2 * idx + 1];
        else if (filter == MISS)
            event = eventSample[socket][2 * idx];
        else if (filter == HIT)
            event = eventSample[socket][2 * idx + 1];

    return event;
}

uint64 PurleyPlatform::getReadBw(uint socket, eventFilter filter)
{
    uint64 readBw = event(socket, filter, PCIRdCur) +
                    event(socket, filter, RFO) +
                    event(socket, filter, CRd) +
                    event(socket, filter, DRd);

    // return (readBw * 64ULL);
    return readBw;
}

uint64 PurleyPlatform::getReadBw()
{
    uint64 readBw = 0;
    for (uint socket = 0; socket < m_socketCount; socket++)
        readBw += (event(socket, TOTAL, PCIRdCur) +
                   event(socket, TOTAL, RFO) +
                   event(socket, TOTAL, CRd) +
                   event(socket, TOTAL, DRd));
    // return (readBw * 64ULL);
    return readBw;
}

uint64 PurleyPlatform::getWriteBw(uint socket, eventFilter filter)
{
    uint64 writeBw = event(socket, filter, RFO) +
                     event(socket, filter, ItoM);
    // return (writeBw * 64ULL);
    return writeBw;
}

uint64 PurleyPlatform::getWriteBw()
{
    uint64 writeBw = 0;
    for (uint socket = 0; socket < m_socketCount; socket++)
        writeBw += (event(socket, TOTAL, RFO) +
                    event(socket, TOTAL, ItoM));
    // return (writeBw * 64ULL);
    return writeBw;
}

//BDX, HSX
class GrantleyPlatform: public LegacyPlatform
{
public:
    GrantleyPlatform(PCM *m, bool csv, bool bandwidth, bool verbose, uint32 delay) :
        LegacyPlatform( {"PCIRdCur", "RFO", "CRd", "DRd","ItoM", "PRd", "WiL"},
                        {
                            {0x19e10000}, //PCIRdCur_miss
                            {0x19e00000}, //PCIRdCur_total
                            {0x18030000}, //RFO_miss
                            {0x18020000}, //RFO_total
                            {0x18110000}, //CRd_miss
                            {0x18100000}, //CRd_total
                            {0x18210000}, //DRd_miss
                            {0x18200000}, //DRd_total
                            {0x1c830000}, //ItoM_miss
                            {0x1c820000}, //ItoM_total
                            {0x18710000}, //PRd_miss
                            {0x18700000}, //PRd_total
                            {0x18f10000}, //WiL_miss
                            {0x18f00000}, //WiL_total
                        },
                        m, csv, bandwidth, verbose, delay)
    {
    };

private:
    enum eventIdx {
        PCIRdCur,
        RFO,
        CRd,
        DRd,
        ItoM,
        PRd,
        WiL,
    };

    enum Events {
            PCIRdCur_miss,
            PCIRdCur_total,
            RFO_miss,
            RFO_total,
            CRd_miss,
            CRd_total,
            DRd_miss,
            DRd_total,
            ItoM_miss,
            ItoM_total,
            PRd_miss,
            PRd_total,
            WiL_miss,
            WiL_total,
    };

    virtual uint64 getReadBw(uint socket, eventFilter filter);
    virtual uint64 getWriteBw(uint socket, eventFilter filter);
    virtual uint64 getReadBw();
    virtual uint64 getWriteBw();
    virtual uint64 event(uint socket, eventFilter filter, uint idx);
};

uint64 GrantleyPlatform::event(uint socket, eventFilter filter, uint idx)
{
    uint64 event = 0;
    if(filter == HIT)
        if (eventSample[socket][2 * idx] < eventSample[socket][2 * idx + 1])
            event = eventSample[socket][2 * idx + 1] - eventSample[socket][2 * idx];
        else
            event = 0;
    else if (filter == MISS)
        event = eventSample[socket][2 * idx];
    else if (filter == TOTAL)
        event = eventSample[socket][2 * idx + 1];

    return event;
}

uint64 GrantleyPlatform::getReadBw(uint socket, eventFilter filter)
{
    uint64 readBw = event(socket, filter, PCIRdCur) +
                    event(socket, filter, RFO) +
                    event(socket, filter, CRd) +
                    event(socket, filter, DRd);

    return (readBw * 64ULL);
}

uint64 GrantleyPlatform::getReadBw()
{
    uint64 readBw = 0;
    for (uint socket = 0; socket < m_socketCount; socket++)
        readBw += (event(socket, TOTAL, PCIRdCur) +
                   event(socket, TOTAL, RFO) +
                   event(socket, TOTAL, CRd) +
                   event(socket, TOTAL, DRd));
    return (readBw * 64ULL);
}

uint64 GrantleyPlatform::getWriteBw(uint socket, eventFilter filter)
{
    uint64 writeBw = event(socket, filter, RFO) +
                     event(socket, filter, ItoM);
    return (writeBw * 64ULL);
}

uint64 GrantleyPlatform::getWriteBw()
{
    uint64 writeBw = 0;
    for (uint socket = 0; socket < m_socketCount; socket++)
        writeBw += (event(socket, TOTAL, RFO) +
                    event(socket, TOTAL, ItoM));
    return (writeBw * 64ULL);
}

//IVT, JKT
class BromolowPlatform: public LegacyPlatform
{
public:
    BromolowPlatform(PCM *m, bool csv, bool bandwidth, bool verbose, uint32 delay) :
        LegacyPlatform( {"PCIeRdCur", "PCIeNSRd", "PCIeWiLF", "PCIeItoM","PCIeNSWr", "PCIeNSWrF"},
                        {
                            {0x19e10000}, //PCIeRdCur_miss
                            {0x19e00000}, //PCIeRdCur_total
                            {0x1e410000}, //PCIeNSRd_miss
                            {0x1e400000}, //PCIeNSRd_total
                            {0x19410000}, //PCIeWiLF_miss
                            {0x19400000}, //PCIeWiLF_total
                            {0x19c10000}, //PCIeItoM_miss
                            {0x19c00000}, //PCIeItoM_total
                            {0x1e510000}, //PCIeNSWr_miss
                            {0x1e500000}, //PCIeNSWr_total
                            {0x1e610000}, //PCIeNSWrF_miss
                            {0x1e600000}, //PCIeNSWrF_total
                        },
                        m, csv, bandwidth, verbose, delay)
    {
    };

private:
    enum eventIdx {
        PCIeRdCur,
        PCIeNSRd,
        PCIeWiLF,
        PCIeItoM,
        PCIeNSWr,
        PCIeNSWrF,
    };

    enum Events {
            PCIeRdCur_miss,
            PCIeRdCur_total,
            PCIeNSRd_miss,
            PCIeNSRd_total,
            PCIeWiLF_miss,
            PCIeWiLF_total,
            PCIeItoM_miss,
            PCIeItoM_total,
            PCIeNSWr_miss,
            PCIeNSWr_total,
            PCIeNSWrF_miss,
            PCIeNSWrF_total,
    };

    virtual uint64 getReadBw(uint socket, eventFilter filter);
    virtual uint64 getWriteBw(uint socket, eventFilter filter);
    virtual uint64 getReadBw();
    virtual uint64 getWriteBw();
    virtual uint64 event(uint socket, eventFilter filter, uint idx);
};

uint64 BromolowPlatform::event(uint socket, eventFilter filter, uint idx)
{
    uint64 event = 0;
    if(filter == HIT)
        if (eventSample[socket][2 * idx] < eventSample[socket][2 * idx + 1])
            event = eventSample[socket][2 * idx + 1] - eventSample[socket][2 * idx];
        else
            event = 0;
    else if (filter == MISS)
        event = eventSample[socket][2 * idx];
    else if (filter == TOTAL)
        event = eventSample[socket][2 * idx + 1];

    return event;
}

uint64 BromolowPlatform::getReadBw(uint socket, eventFilter filter)
{
    uint64 readBw = event(socket, filter, PCIeRdCur) +
                    event(socket, filter, PCIeNSWr);
    return (readBw * 64ULL);
}

uint64 BromolowPlatform::getReadBw()
{
    uint64 readBw = 0;
    for (uint socket = 0; socket < m_socketCount; socket++)
        readBw += (event(socket, TOTAL, PCIeRdCur) +
                   event(socket, TOTAL, PCIeNSWr));
    return (readBw * 64ULL);
}

uint64 BromolowPlatform::getWriteBw(uint socket, eventFilter filter)
{
    uint64 writeBw = event(socket, filter, PCIeWiLF) +
                     event(socket, filter, PCIeItoM) +
                     event(socket, filter, PCIeNSWr) +
                     event(socket, filter, PCIeNSWrF);
    return (writeBw * 64ULL);
}

uint64 BromolowPlatform::getWriteBw()
{
    uint64 writeBw = 0;
    for (uint socket = 0; socket < m_socketCount; socket++)
        writeBw += (event(socket, TOTAL, PCIeWiLF) +
                    event(socket, TOTAL, PCIeItoM) +
                    event(socket, TOTAL, PCIeNSWr) +
                    event(socket, TOTAL, PCIeNSWrF));
    return (writeBw * 64ULL);
}