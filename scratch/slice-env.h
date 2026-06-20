#pragma once

#if __has_include("ns3/opengym-module.h")
#define HAVE_OPENGYM

#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/network-module.h"
#include "ns3/nr-module.h"
#include "ns3/opengym-module.h"

#include <array>
#include <limits>
#include <unordered_map>

namespace ns3
{

class NrSliceGymEnv : public OpenGymEnv
{
  public:
    static constexpr uint32_t kSliceCount  = 3;
    static constexpr uint32_t kObsSize     = 18;
    static constexpr uint32_t kActionCount = 27;

    enum SliceId : uint8_t
    {
        EMBB  = 0,
        URLLC = 1,
        MMTC  = 2
    };


    struct Config
    {
        uint16_t totalPrbs{51};
        Time stepInterval{MilliSeconds(100)};
        Time simTime{Seconds(100)};
        std::array<uint16_t, kSliceCount> initialPrbAlloc{25, 15, 11};


        std::array<uint32_t, kSliceCount> maxUes{10, 5, 5};


        std::array<double, kSliceCount> maxThrMbps{130.0, 30.0, 8.0};

        std::array<double, kSliceCount> maxLatMs{30.0,  10.0, 150.0};
        std::array<double, kSliceCount> minThrMbps{70.0,  2.5,  0.1};
    };

    NrSliceGymEnv();
    ~NrSliceGymEnv() override;

    static TypeId GetTypeId();

    void Initialize(const Config& cfg,
                    const Ptr<NrHelper>& nrHelper,
                    const NetDeviceContainer& gnbDevs,
                    const std::array<NetDeviceContainer, kSliceCount>& ueDevsBySlice);

    void SetFlowMonitor(const Ptr<FlowMonitor>& flowMonitor,
                        const Ptr<Ipv4FlowClassifier>& flowClassifier);

    void BuildImsiSliceMap();

    void OnSchedulerNotify(
        const std::vector<NrMacSchedulerUeInfoAi::LcObservation>& observations,
        bool isGameOver,
        float reward,
        const std::string& extraInfo,
        const NrMacSchedulerUeInfoAi::UpdateAllUeWeightsFn& updateWeightsFn);

    // OpenGymEnv API
    Ptr<OpenGymSpace>      GetActionSpace()      override;
    Ptr<OpenGymSpace>      GetObservationSpace() override;
    bool                   GetGameOver()         override;
    Ptr<OpenGymDataContainer> GetObservation()   override;
    float                  GetReward()           override;
    std::string            GetExtraInfo()        override;
    bool ExecuteActions(Ptr<OpenGymDataContainer> action) override;

  private:
    void ScheduleStep();
    void AggregateFlowStats();
    void AggregateHolDelay();
    void ApplySliceWeights();
    void EnforceConstraints();

    uint8_t ResolveSliceFromPort(uint16_t port) const;
    uint8_t ResolveSliceFromAddress(const Ipv4Address& dst) const;
    void    TryBuildRntiSliceMap();
    
    Config m_cfg{};

    Ptr<NrHelper>         m_nrHelper;
    NetDeviceContainer    m_gnbDevs;
    std::array<NetDeviceContainer, kSliceCount> m_ueDevsBySlice;

    Ptr<FlowMonitor>        m_flowMonitor;
    Ptr<Ipv4FlowClassifier> m_flowClassifier;

    std::unordered_map<uint64_t, uint8_t> m_imsiToSlice;
    std::unordered_map<Ipv4Address, uint8_t, Ipv4AddressHash> m_ipToSlice;
    std::unordered_map<uint16_t, uint8_t> m_rntiToSlice;
    std::unordered_map<uint32_t, uint64_t> m_prevRxPackets{};
    std::unordered_map<uint32_t, double>   m_prevDelaySum{};

    bool     m_rntiMapReady{false};
    bool     m_gameOver{false};
    bool     m_initialized{false};
    uint32_t m_stepCount{0};

    float       m_reward{0.0F};
    std::string m_extraInfo;

    NrMacSchedulerUeInfoAi::UeWeightsMap m_weightsMap;
    NrMacSchedulerUeInfoAi::UpdateAllUeWeightsFn m_updateWeightsFn;
    std::vector<NrMacSchedulerUeInfoAi::LcObservation> m_lastLcObservations;

    std::array<uint16_t, kSliceCount> m_prbAlloc{10, 8, 7};
    std::array<float,    kObsSize>    m_observation{};
    std::array<uint32_t, kSliceCount> m_backlogBytes{};
    std::array<uint32_t, kSliceCount> m_backlogAccum{};
    std::array<double, kSliceCount> m_thrMbps{};
    std::array<double, kSliceCount> m_latMs{};
    std::array<double, kSliceCount> m_holNorm{};      // HOL delay normalised to [0,1]
    std::array<double, kSliceCount> m_holSumMs{};
    std::array<uint32_t, kSliceCount > m_holSamples{};
    std::array<uint32_t, kSliceCount> m_uesPerSlice{};
    std::array<double, kSliceCount> m_servedDemandRatio{};
    std::array<uint8_t, kSliceCount> m_demandActive{};
    std::array<bool, kSliceCount> m_schedulerActiveThisStep{};
    std::array<bool, kSliceCount> m_latValid{};
    
    
    std::unordered_map<uint32_t, uint64_t> m_prevRxBytes{};

    std::unordered_map<uint32_t, uint64_t> m_prevLostPackets;


    std::array<double, kSliceCount> m_pktLossRate{0.0, 0.0, 0.0};
};

} // namespace ns3

#endif // HAVE_OPENGYM
