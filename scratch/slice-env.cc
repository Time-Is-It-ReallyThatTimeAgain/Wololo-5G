#include "slice-env.h"

#ifdef HAVE_OPENGYM

#include "ns3/integer.h"
#include "ns3/nr-ue-mac.h"
#include "ns3/nr-ue-net-device.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("NrSliceGymEnv");
NS_OBJECT_ENSURE_REGISTERED(NrSliceGymEnv);

namespace
{
constexpr std::array<std::array<int8_t, NrSliceGymEnv::kSliceCount>, NrSliceGymEnv::kActionCount>
    kActionDelta = {{{-1, -1, -1},
                     {-1, -1,  0},
                     {-1, -1,  1},
                     {-1,  0, -1},
                     {-1,  0,  0},
                     {-1,  0,  1},
                     {-1,  1, -1},
                     {-1,  1,  0},
                     {-1,  1,  1},
                     { 0, -1, -1},
                     { 0, -1,  0},
                     { 0, -1,  1},
                     { 0,  0, -1},
                     { 0,  0,  0},
                     { 0,  0,  1},
                     { 0,  1, -1},
                     { 0,  1,  0},
                     { 0,  1,  1},
                     { 1, -1, -1},
                     { 1, -1,  0},
                     { 1, -1,  1},
                     { 1,  0, -1},
                     { 1,  0,  0},
                     { 1,  0,  1},
                     { 1,  1, -1},
                     { 1,  1,  0},
                     { 1,  1,  1}}};

float
Clamp01(double v)
{
    return static_cast<float>(std::max(0.0, std::min(1.0, v)));
}
constexpr std::array<double, NrSliceGymEnv::kSliceCount> kThrW  = {0.70, 0.10, 0.50};
constexpr std::array<double, NrSliceGymEnv::kSliceCount> kLatW  = {0.30, 0.70, 0.50};
constexpr std::array<double, NrSliceGymEnv::kSliceCount> kPelrW = {0.00, 0.20, 0.00};
} // namespace

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

NrSliceGymEnv::NrSliceGymEnv()
{
    m_observation.fill(0.0F);
}

NrSliceGymEnv::~NrSliceGymEnv() = default;

TypeId
NrSliceGymEnv::GetTypeId()
{
    static TypeId tid = TypeId("ns3::NrSliceGymEnv")
                            .SetParent<OpenGymEnv>()
                            .AddConstructor<NrSliceGymEnv>();
    return tid;
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

void
NrSliceGymEnv::Initialize(const Config& cfg,
                          const Ptr<NrHelper>& nrHelper,
                          const NetDeviceContainer& gnbDevs,
                          const std::array<NetDeviceContainer, kSliceCount>& ueDevsBySlice)
{
    m_cfg           = cfg;
    m_nrHelper      = nrHelper;
    m_gnbDevs       = gnbDevs;
    m_ueDevsBySlice = ueDevsBySlice;
    m_prbAlloc      = cfg.initialPrbAlloc;

    m_uesPerSlice = {
        ueDevsBySlice[EMBB].GetN(),
        ueDevsBySlice[URLLC].GetN(),
        ueDevsBySlice[MMTC].GetN()
    };

    BuildImsiSliceMap();
    m_weightsMap.clear();
    const double defaultWeightInit =
        1.0 / static_cast<double>(std::max<uint32_t>(1, cfg.totalPrbs));
    for (uint16_t r = 1; r <= std::numeric_limits<uint8_t>::max(); ++r)
    {
        auto& ueWeights = m_weightsMap[static_cast<uint8_t>(r)];
        for (uint16_t lc = 1; lc <= std::numeric_limits<uint8_t>::max(); ++lc)
        {
            ueWeights[static_cast<uint8_t>(lc)] = static_cast<float>(defaultWeightInit);
        }
    }
    m_prevRxBytes.clear();
    m_prevRxPackets.clear();
    m_prevDelaySum.clear();
    m_prevLostPackets.clear();

    m_initialized = true;
    m_stepCount   = 0;
    Simulator::Schedule(m_cfg.stepInterval, &NrSliceGymEnv::ScheduleStep, this);
}

void
NrSliceGymEnv::SetFlowMonitor(const Ptr<FlowMonitor>& flowMonitor,
                              const Ptr<Ipv4FlowClassifier>& flowClassifier)
{
    m_flowMonitor    = flowMonitor;
    m_flowClassifier = flowClassifier;
}

// ---------------------------------------------------------------------------
// RNTI / IMSI map construction
// ---------------------------------------------------------------------------

void
NrSliceGymEnv::BuildImsiSliceMap()
{
    m_imsiToSlice.clear();

    for (uint8_t s = 0; s < kSliceCount; ++s)
    {
        for (uint32_t i = 0; i < m_ueDevsBySlice[s].GetN(); ++i)
        {
            Ptr<NrUeNetDevice> ueDev =
                DynamicCast<NrUeNetDevice>(m_ueDevsBySlice[s].Get(i));
            if (!ueDev)
            {
                continue;
            }
            m_imsiToSlice[ueDev->GetImsi()] = s;
        }
    }
}

void
NrSliceGymEnv::TryBuildRntiSliceMap()
{
    if (m_rntiMapReady)
    {
        return;
    }

    m_rntiToSlice.clear();
    bool allResolved = true;

    for (uint8_t s = 0; s < kSliceCount; ++s)
    {
        for (uint32_t i = 0; i < m_ueDevsBySlice[s].GetN(); ++i)
        {
            Ptr<NrUeNetDevice> ueDev =
                DynamicCast<NrUeNetDevice>(m_ueDevsBySlice[s].Get(i));
            if (!ueDev || !ueDev->GetMac(0))
            {
                allResolved = false;
                continue;
            }

            const uint16_t rnti = ueDev->GetMac(0)->GetRnti();
            if (rnti == std::numeric_limits<uint16_t>::max())
            {
                allResolved = false;
                continue;
            }

            m_rntiToSlice[rnti] = s;
        }
    }

    m_rntiMapReady = allResolved && !m_rntiToSlice.empty();
}

// ---------------------------------------------------------------------------
// Scheduler callback
// ---------------------------------------------------------------------------

void
NrSliceGymEnv::OnSchedulerNotify(
    const std::vector<NrMacSchedulerUeInfoAi::LcObservation>& observations,
    bool isGameOver,
    float reward,
    const std::string& extraInfo,
    const NrMacSchedulerUeInfoAi::UpdateAllUeWeightsFn& updateWeightsFn)
{
    m_lastLcObservations = observations;
    m_gameOver           = isGameOver;
    m_reward             = reward;
    m_extraInfo          = extraInfo;
    m_updateWeightsFn    = updateWeightsFn;

    TryBuildRntiSliceMap();             
    if (!m_rntiMapReady)
    {
        NS_LOG_WARN("OnSchedulerNotify: RNTI map not ready — dropping "
                    << observations.size() << " HOL observations this callback.");
    }

    if (m_rntiMapReady)      
    {
        for ( const auto & obs : observations)
 
        {
            auto it = m_rntiToSlice.find(obs.rnti);
            if (it == m_rntiToSlice.end())
            {
                continue;
            }
            const uint8_t slice = it->second;
 
            if (slice >= kSliceCount)
            {
                continue;
            }
            m_holSumMs[slice] += static_cast <double>(obs.holDelay);
            ++m_holSamples[slice];
            m_backlogAccum[slice] += obs.pendingBytes;
        }
    }
    ApplySliceWeights();
}

// ---------------------------------------------------------------------------
// OpenGymEnv API
// ---------------------------------------------------------------------------

Ptr<OpenGymSpace>
NrSliceGymEnv::GetActionSpace()
{
    return CreateObject<OpenGymDiscreteSpace>(kActionCount);
}

Ptr<OpenGymSpace>
NrSliceGymEnv::GetObservationSpace()
{
    const float low  = 0.0F;
    const float high = 1.0F;
    std::vector<uint32_t> shape = {kObsSize};
    return CreateObject<OpenGymBoxSpace>(low, high, shape, TypeNameGet<float>());
}

bool
NrSliceGymEnv::GetGameOver()
{
    return m_gameOver;
}

Ptr<OpenGymDataContainer>
NrSliceGymEnv::GetObservation()
{
    std::vector<uint32_t> shape = {kObsSize};
    Ptr<OpenGymBoxContainer<float>> box =
        CreateObject<OpenGymBoxContainer<float>>(shape);

    for (uint32_t i = 0; i < kObsSize; ++i)
    {
        box->AddValue(m_observation[i]);
    }

    return box;
}

float
NrSliceGymEnv::GetReward()
{
    return m_reward;
}

std::string
NrSliceGymEnv::GetExtraInfo()
{
    return m_extraInfo;
}

bool
NrSliceGymEnv::ExecuteActions(Ptr<OpenGymDataContainer> action)
{
    Ptr<OpenGymDiscreteContainer> discrete =
        DynamicCast<OpenGymDiscreteContainer>(action);
    if (!discrete)
    {
        NS_LOG_WARN("NrSliceGymEnv received non-discrete action container");
        return false;
    }

    const uint32_t actionId = static_cast<uint32_t>(discrete->GetValue());
    if (actionId >= kActionCount)
    {
        NS_LOG_WARN("Action id out of range: " << actionId);
        return false;
    }

    for (uint32_t s = 0; s < kSliceCount; ++s)
    {
        const int32_t updated =
            static_cast<int32_t>(m_prbAlloc[s]) + kActionDelta[actionId][s];
        m_prbAlloc[s] = static_cast<uint16_t>(std::max(1, updated));
    }

    EnforceConstraints();
    ApplySliceWeights();

    return true;
}

// ---------------------------------------------------------------------------
// EnforceConstraints — keeps sum(m_prbAlloc) == totalPrbs with min=1 per slice
// ---------------------------------------------------------------------------

void
NrSliceGymEnv::EnforceConstraints()
{
    for (auto& prb : m_prbAlloc)
    {
        prb = std::max<uint16_t>(1, prb);
    }

    int32_t diff = static_cast<int32_t>(m_cfg.totalPrbs) -
                   static_cast<int32_t>(m_prbAlloc[0] + m_prbAlloc[1] + m_prbAlloc[2]);

    while (diff > 0)
    {
        const uint8_t minSlice = static_cast<uint8_t>(
            std::distance(m_prbAlloc.begin(),
                          std::min_element(m_prbAlloc.begin(), m_prbAlloc.end())));
        ++m_prbAlloc[minSlice];
        --diff;
    }

    int safetyCounter = 0;
    while (diff < 0)
    {
        bool removed = false;
        for (uint8_t i = 0; i < kSliceCount && diff < 0; ++i)
        {
            const uint8_t idx = (m_stepCount + i) % kSliceCount;
            if (m_prbAlloc[idx] > 1)
            {
                --m_prbAlloc[idx];
                ++diff;
                removed = true;
            }
        }
        if (!removed || ++safetyCounter > 100)
        {
            NS_LOG_WARN("NrSliceGymEnv::EnforceConstraints: "
                        "cannot converge, resetting to initialPrbAlloc");
            m_prbAlloc = m_cfg.initialPrbAlloc;
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// ApplySliceWeights — pushes PRB-proportional weights to the AI scheduler
// ---------------------------------------------------------------------------

void
NrSliceGymEnv::ApplySliceWeights()
{
    if (!m_updateWeightsFn)
    {
        return;
    }

    TryBuildRntiSliceMap();

    const double defaultWeight =
        1.0 / static_cast<double>(
            std::max<uint32_t>(1, m_cfg.totalPrbs));

    
     
    for (const auto& [rnti16, slice] : m_rntiToSlice)
    {
        if (rnti16 > 255)
        {
            NS_LOG_ERROR(
                "ApplySliceWeights: RNTI "
                << rnti16
                << " exceeds uint8_t range.");

            continue;
        }

        double weight = defaultWeight;

        if (slice < kSliceCount)
        {
            const double sliceFrac =
                static_cast<double>(m_prbAlloc[slice]) /
                static_cast<double>(m_cfg.totalPrbs);

            const double ueCount =
                static_cast<double>(
                    std::max<uint32_t>(
                        1,
                        m_uesPerSlice[slice]));

            weight = sliceFrac / ueCount;
        }

        auto& ueWeights =
            m_weightsMap[static_cast<uint8_t>(rnti16)];

        for (uint16_t lc = 1; lc <= std::numeric_limits<uint8_t>::max(); ++lc)
        {
            ueWeights[static_cast<uint8_t>(lc)] =
                static_cast<float>(weight);
        }
    }


  
    for (const auto& obs : m_lastLcObservations)
    {
        const uint16_t rnti16 = obs.rnti;
        const uint8_t  lcId   = obs.lcId;

        if (lcId == 0)
        {
            continue;
        }

        if (rnti16 > 255)
        {
            NS_LOG_ERROR(
                "ApplySliceWeights: RNTI "
                << rnti16
                << " exceeds uint8_t range.");

            continue;
        }

        auto& ueWeights =
            m_weightsMap[static_cast<uint8_t>(rnti16)];

        auto it = m_rntiToSlice.find(rnti16);

        if (it == m_rntiToSlice.end())
        {
           
            ueWeights[lcId] =
                static_cast<float>(defaultWeight);

            continue;
        }

        const uint8_t slice = it->second;

        if (slice >= kSliceCount)
        {
            ueWeights[lcId] =
                static_cast<float>(defaultWeight);

            continue;
        }

        const double sliceFrac =
            static_cast<double>(m_prbAlloc[slice]) /
            static_cast<double>(m_cfg.totalPrbs);

        const double ueCount =
            static_cast<double>(
                std::max<uint32_t>(
                    1,
                    m_uesPerSlice[slice]));

        const double weight =
            sliceFrac / ueCount;

        ueWeights[lcId] =
            static_cast<float>(weight);
    }

    if (!m_weightsMap.empty())
    {
        m_updateWeightsFn(m_weightsMap);
    }
}

void
NrSliceGymEnv::AggregateHolDelay()
{   
    m_schedulerActiveThisStep.fill(false);
    TryBuildRntiSliceMap();
    if (!m_rntiMapReady)
    {
        return;   
    }



    for (uint8_t s = 0; s < kSliceCount; ++s)
    {   m_schedulerActiveThisStep[s] = (m_holSamples[s] > 0);
        if (m_holSamples [s] == 0 )
        {
            continue;
        }

        m_holNorm[s] = Clamp01(meanHolMs / std::max(1e-9, 2.0 * m_cfg.maxLatMs[s]));
        m_holSumMs[s] = 0.0;
        m_holSamples[s] = 0;
        
    }
}

uint8_t
NrSliceGymEnv::ResolveSliceFromPort(uint16_t port) const
{
    if (port >= 1000 && port <= 1999) return EMBB;
    if (port >= 2000 && port <= 2999) return URLLC;
    if (port >= 3000 && port <= 3999) return MMTC;

    
    return kSliceCount;  
}

uint8_t
NrSliceGymEnv::ResolveSliceFromAddress(const Ipv4Address& dst) const
{
    auto it = m_ipToSlice.find(dst);
    if (it == m_ipToSlice.end())
    {
        return kSliceCount;
    }
    return it->second;
}

void
NrSliceGymEnv::AggregateFlowStats()
{
    m_thrMbps.fill(0.0);
    m_pktLossRate.fill(0.0);
    m_latValid.fill(false);

    if (!m_flowMonitor || !m_flowClassifier)
    {
        return;
    }

    const auto stats = m_flowMonitor->GetFlowStats();

    std::array<double,   kSliceCount> totalDeltaDelayMs{0.0, 0.0, 0.0};
    std::array<uint64_t, kSliceCount> totalDeltaPkts   {0,   0,   0  };
    std::array<uint64_t, kSliceCount> totalDeltaLost   {0,   0,   0  };

    for (const auto& [flowId, st] : stats)
    {
        Ipv4FlowClassifier::FiveTuple fiveTuple =
            m_flowClassifier->FindFlow(flowId);

        uint8_t slice = ResolveSliceFromAddress(fiveTuple.destinationAddress);
            if (slice >= kSliceCount)
            {
             slice = ResolveSliceFromPort(fiveTuple.destinationPort);
            }

        if (slice >= kSliceCount)
        {
            continue;   
        }


        const uint64_t prevBytes  = m_prevRxBytes.count(flowId)
                                  ? m_prevRxBytes[flowId] : 0;
        const uint64_t deltaBytes = (st.rxBytes >= prevBytes)
                                  ? (st.rxBytes - prevBytes) : st.rxBytes;
        m_prevRxBytes[flowId]     = st.rxBytes;

        m_thrMbps[slice] +=
            (static_cast<double>(deltaBytes) * 8.0 / 1e6) /
            std::max(1e-9, m_cfg.stepInterval.GetSeconds());

        
        const uint64_t prevPkts  = m_prevRxPackets.count(flowId)
                                 ? m_prevRxPackets[flowId] : 0;
        const double   prevDelay = m_prevDelaySum.count(flowId)
                                 ? m_prevDelaySum[flowId]  : 0.0;
        const uint64_t deltaPkts  = (st.rxPackets >= prevPkts)
                                  ? (st.rxPackets - prevPkts) : 0;
        const double   deltaDelay = (st.delaySum.GetSeconds() >= prevDelay)
                                  ? (st.delaySum.GetSeconds() - prevDelay) : 0.0;

        m_prevRxPackets[flowId] = st.rxPackets;
        m_prevDelaySum[flowId]  = st.delaySum.GetSeconds();

        if (deltaPkts > 0)
        {
            totalDeltaDelayMs[slice] += deltaDelay * 1e3;
            totalDeltaPkts[slice]    += deltaPkts;
        }

        const uint64_t prevLost  = m_prevLostPackets.count(flowId)
                                 ? m_prevLostPackets[flowId] : 0;
        const uint64_t deltaLost = (st.lostPackets >= prevLost)
                                 ? (st.lostPackets - prevLost) : 0;
        m_prevLostPackets[flowId]  = st.lostPackets;
        totalDeltaLost[slice]     += deltaLost;
    }

    for (uint8_t s = 0; s < kSliceCount; ++s)
    {
        if (totalDeltaPkts[s] > 0)
        {
            m_latMs[s] = totalDeltaDelayMs[s] /static_cast<double>(totalDeltaPkts[s]);
            m_latValid[s] = true;
        }

        const uint64_t totalObsPkts = totalDeltaLost[s] + totalDeltaPkts[s];
        if (totalObsPkts > 0)
        {
            m_pktLossRate[s] = static_cast<double>(totalDeltaLost[s])/static_cast<double>(totalObsPkts);
        }
    }
}

// ---------------------------------------------------------------------------
// Scheduled step — observation + reward + notify
// ---------------------------------------------------------------------------

void
NrSliceGymEnv::ScheduleStep()
{
    if (!m_initialized)
    {
        return;
    }

    ++m_stepCount;
    AggregateFlowStats();
    AggregateHolDelay();
    m_backlogBytes = m_backlogAccum;
    m_backlogAccum.fill(0);

    for (uint8_t s = 0; s < kSliceCount; ++s)
    {
        if (m_thrMbps[s] <= 0.001 && m_backlogBytes[s] == 0)
            m_latMs[s] = 0.0;

    }


    double totalBacklogObs = 0.0;
    
    for (uint8_t s = 0; s < kSliceCount; ++s)
    {
        totalBacklogObs += static_cast<double>(m_backlogBytes[s]);
    }

    const bool useBacklogObs = (totalBacklogObs > 0.0);
    double totalDemandObs = totalBacklogObs;
    if (!useBacklogObs)
    {
        totalDemandObs = 0.0;
        for (uint8_t s = 0; s < kSliceCount; ++s)
        {
            if (m_thrMbps[s] > 0.001)
            {
                totalDemandObs += m_thrMbps[s];
            }
        }
    }
    for (uint8_t s = 0; s < kSliceCount; ++s)
    {
        m_observation[s]      = Clamp01(static_cast<double>(m_prbAlloc[s])/m_cfg.totalPrbs);
        m_observation[3 + s]  = Clamp01(m_thrMbps[s] / m_cfg.maxThrMbps[s]);
        m_observation[6 + s]  = Clamp01(m_latMs[s]/(2.0 * m_cfg.maxLatMs[s]));
        m_observation[9 + s]  = m_holNorm[s];
        {
            const double prbFrac    = static_cast<double>(m_prbAlloc[s]) /
                                      static_cast<double>(m_cfg.totalPrbs);
            const double rawDemand  = useBacklogObs
                ? static_cast<double>(m_backlogBytes[s])
                : m_thrMbps[s];
            const double demandFrac = (totalDemandObs > 1e-9)
                ? (rawDemand / totalDemandObs)
                : 0.0;
            const double align      = (totalDemandObs > 1e-9)
                ? (1.0 - std::abs(prbFrac - demandFrac))
                : 0.0;
            m_observation[12 + s]  = Clamp01(align);
        }

        if (s == URLLC)
            m_observation[15 + s] = Clamp01(m_pktLossRate[URLLC] / (2.0 * 1e-4));
        else
            m_observation[15 + s] = Clamp01(m_thrMbps[s] / m_cfg.minThrMbps[s]);

    }    

    double   thrNormAvg    = 0.0;
    double   slaMarginAvg  = 0.0;
    uint32_t slaViolations = 0;
    double   totalThrMbps  = 0.0;   
    uint32_t activeSlices  = 0;

    m_servedDemandRatio.fill(0.0);
    m_demandActive.fill(0);


    for (uint8_t s = 0; s < kSliceCount; ++s)
    {
        const double thrNorm = m_thrMbps[s] / std::max(1e-9, m_cfg.maxThrMbps[s]);
        const bool demandActive = (m_thrMbps[s] > 0.001) || (m_backlogBytes[s] > 0);

        m_demandActive[s] = demandActive ? 1 : 0;

        if (!demandActive)
        {
            continue;
        }
    
        const bool latSlaOk = (m_latMs[s] <= m_cfg.maxLatMs[s]);

        const bool pelrSlaOk = (s != URLLC) || (m_pktLossRate[URLLC] <= 1e-4);

        const bool slaSat =
            (m_thrMbps[s] >= m_cfg.minThrMbps[s]) && latSlaOk && pelrSlaOk;

     
        const double thrMargin = std::tanh(std::max(0.0,
            (m_thrMbps[s] - m_cfg.minThrMbps[s]) / std::max(1e-9, m_cfg.minThrMbps[s])));

        const double latMargin = (s == URLLC)
            ? std::tanh(std::max(0.0, (0.5 - m_holNorm[s]) / 0.5))
            : std::tanh(std::max(0.0,
                (m_cfg.maxLatMs[s] - m_latMs[s]) / std::max(1e-9, m_cfg.maxLatMs[s])));

        slaViolations += slaSat ? 0 : 1;

        const double thrNormSoft = std::tanh(std::max(0.0, thrNorm));
        thrNormAvg   += thrNormSoft;


        const double pelrMargin = (s == URLLC)
            ? std::tanh(std::max(0.0, (1e-4 - m_pktLossRate[s]) / 1e-4))
            : 0.0;


        slaMarginAvg += kThrW[s] * thrMargin + kLatW[s] * latMargin + kPelrW[s] * pelrMargin;
        totalThrMbps += m_thrMbps[s];
        ++activeSlices;

        m_servedDemandRatio[s] = std::min(2.0, m_thrMbps[s] / std::max(1e-9, m_cfg.minThrMbps[s]));
    }

        double slaMarginNorm = 0.0;
        double effNorm       = 0.0;
        double violationRate = 0.0;
    

        if (activeSlices == 0)
    {

        m_reward = 0.0F;
    }
        else
    {
        const uint32_t n         = activeSlices;  
        thrNormAvg              /= n;
        slaMarginAvg            /= n;
    
        slaMarginNorm = slaMarginAvg ;          

        const bool useBacklogEff = (totalBacklogObs > 0.0);
        const double totalDemandEff = useBacklogEff ? totalBacklogObs : totalThrMbps;
        if (totalDemandEff > 1e-9)
        {
            double alignScore = 0.0;
            for (uint8_t s = 0; s < kSliceCount; ++s)
            {
                if (m_demandActive[s] == 0) { continue; }
                const double rawDemand  = useBacklogEff
                    ? static_cast<double>(m_backlogBytes[s])
                    : m_thrMbps[s];
                const double demandFrac = rawDemand / totalDemandEff;

                const double prbFracS   = static_cast<double>(m_prbAlloc[s]) /
                                          static_cast<double>(m_cfg.totalPrbs);
                alignScore += 1.0 - std::abs(prbFracS - demandFrac);
            }
            effNorm = alignScore / static_cast<double>(n);
        }
      

        violationRate = static_cast<double>(slaViolations) /                             
                                 static_cast<double>(n);

        m_reward = static_cast<float>(
        0.451 * thrNormAvg    +
        0.292 * slaMarginNorm +
        0.257 * effNorm       -
        1.20  * violationRate);

    }

             m_extraInfo =
        std::string("{") +
     
        "\"demand_active\":["        + std::to_string(m_demandActive[0])      + "," +
                                       std::to_string(m_demandActive[1])      + "," +
                                       std::to_string(m_demandActive[2])      + "]," +
        "\"served_demand_ratio\":["  + std::to_string(m_servedDemandRatio[0]) + "," +
                                       std::to_string(m_servedDemandRatio[1]) + "," +
                                       std::to_string(m_servedDemandRatio[2]) + "]," +
      
        "\"lat_ms\":["               + std::to_string(m_latMs[0])             + "," +
                                       std::to_string(m_latMs[1])             + "," +
                                       std::to_string(m_latMs[2])             + "]," +
        "\"lat_valid\":["            + std::to_string(m_latValid[0] ? 1 : 0)  + "," +
                                       std::to_string(m_latValid[1] ? 1 : 0)  + "," +
                                       std::to_string(m_latValid[2] ? 1 : 0)  + "]," +
        "\"hol_norm\":["             + std::to_string(m_holNorm[0])           + "," +
                                       std::to_string(m_holNorm[1])           + "," +
                                       std::to_string(m_holNorm[2])           + "]," +
 
        "\"pkt_loss_rate\":["        + std::to_string(m_pktLossRate[0])       + "," +
                                       std::to_string(m_pktLossRate[1])       + "," +
                                       std::to_string(m_pktLossRate[2])       + "]," +
        "\"urllc_pelr_norm\":"       + std::to_string(
                                       std::max(0.0, std::min(1.0, m_pktLossRate[URLLC] / (2.0 * 1e-4)))) + ","
     
        "\"reward_terms\":{"                                                          +
            "\"thr_norm_avg\":"      + std::to_string(thrNormAvg)             + "," +
            "\"sla_margin_norm\":"   + std::to_string(slaMarginNorm)          + "," +
            "\"eff_norm\":"          + std::to_string(effNorm)                + "," +
            "\"violation_rate\":"    + std::to_string(violationRate)          + "," +
            "\"active_slices\":"     + std::to_string(activeSlices)           +
        "}," +

        "\"sla_lat_gate_signal\":[\"fm\",\"fm\",\"fm\"]," +

        "\"lat_margin_signal\":[\"fm\",\"hol\",\"fm\"]," +
        
        "\"cfg\":{"                                                                    +
            "\"max_thr_mbps\":["     + std::to_string(m_cfg.maxThrMbps[0])    + "," +
                                       std::to_string(m_cfg.maxThrMbps[1])    + "," +
                                       std::to_string(m_cfg.maxThrMbps[2])    + "]," +
            "\"min_thr_mbps\":["     + std::to_string(m_cfg.minThrMbps[0])    + "," +
                                       std::to_string(m_cfg.minThrMbps[1])    + "," +
                                       std::to_string(m_cfg.minThrMbps[2])    + "]," +
            "\"max_lat_ms\":["       + std::to_string(m_cfg.maxLatMs[0])      + "," +
                                       std::to_string(m_cfg.maxLatMs[1])      + "," +
                                       std::to_string(m_cfg.maxLatMs[2])      + "]"  +
        "}"  +
        "}";
    

    m_gameOver = Simulator::Now() >= m_cfg.simTime;

    Notify();

    if (!m_gameOver)
    {
        Simulator::Schedule(m_cfg.stepInterval,
                            &NrSliceGymEnv::ScheduleStep, this);
    }
}

} // namespace ns3

#endif 