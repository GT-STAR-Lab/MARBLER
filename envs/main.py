import warnings

#from PCP import demoPCPAgents, pcpAgents
from PCP_Grid import gridPcpAgents, demoPCPAgents_Grid
#from PCP.pcpAgents import * 



if __name__ == "__main__":
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)

    args = gridPcpAgents.create_parser().parse_args()

    predatorPolicy = demoPCPAgents_Grid.DemoPredatorAgent()
    capturePolicy = demoPCPAgents_Grid.DemoCaptureAgent()
    policies = []

    #Uses the two policies; one for each predator agent and one for each capture agent
    for i in range(args.predator):
        policies.append(predatorPolicy)
    for i in range(args.capture):
        policies.append(capturePolicy)

    agents = gridPcpAgents.PCPAgents(args, policies)
    agents.run_episode()
