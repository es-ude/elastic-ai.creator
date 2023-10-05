from elasticai.creator.file_generation.savable import Path
from elasticai.creator.vhdl.system_integrations.env5_constraints.env5_constraints import ENV5Constraints
from elasticai.creator.vhdl.system_integrations.middleware.middleware import Middleware
from elasticai.creator.vhdl.system_integrations.skeleton.skeleton import LSTMSkeleton
from elasticai.creator.vhdl.system_integrations.top.env5_reconfig_top import ENV5ReconfigTop


class PlugAndPlaySolutionENV5:
    def save_to(self, destination: Path):
        def save_srcs(destination: Path):
            lstm_skeleton = LSTMSkeleton()
            lstm_skeleton.save_to(destination.create_subpath(subpath_name='skeleton'))

            middleware = Middleware()
            middleware.save_to(destination.create_subpath(subpath_name='middleware'))

            env5_reconfig_top = ENV5ReconfigTop()
            env5_reconfig_top.save_to(destination.create_subpath(subpath_name='env5_reconfig_top'))

        def save_constraints(destination: Path):
            env5_config = ENV5Constraints()
            env5_config.save_to(destination.create_subpath(subpath_name='env5_config'))

