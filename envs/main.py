import warnings

from PCP_Cont import demoPCPAgents_Cont, parser_cont
from PCP_Grid import demoPCPAgents_Grid, parser_grid
import pcpAgents



if __name__ == "__main__":
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)

    type = 'cont'

    if type == 'grid':
        args = parser_grid.create_parser().parse_args()

        predatorPolicy = demoPCPAgents_Grid.DemoPredatorAgent()
        capturePolicy = demoPCPAgents_Grid.DemoCaptureAgent()
    else:
        args = parser_cont.create_parser().parse_args()

        predatorPolicy = demoPCPAgents_Cont.DemoPredatorAgent()
        capturePolicy = demoPCPAgents_Cont.DemoCaptureAgent()

    policies = []

    #Uses the two policies; one for each predator agent and one for each capture agent
    for i in range(args.predator):
        policies.append(predatorPolicy)
    for i in range(args.capture):
        policies.append(capturePolicy)

    agents = pcpAgents.PCPAgents(args, policies, type=type)
    for i in range(20):
        agents.run_episode()
    del agents
