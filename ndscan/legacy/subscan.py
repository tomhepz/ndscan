"""
Implements subscans, that is, the ability for an :class:`.ExpFragment` to scan
another child fragment as part of its execution.
"""

import logging
import re
from collections import OrderedDict
from copy import copy
from functools import reduce

from artiq.language import kernel, portable, rpc

from ..utils import merge_no_duplicates, shorten_to_unambiguous_suffixes
from ..define.default_analysis import AnnotationContext, DefaultAnalysis
from ..define.fragment import ExpFragment, Fragment, RestartKernelTransitoryError
from ..define.parameters import ParamHandle
from ..define.result_channels import (
    AppendingDatasetSink,
    ArraySink,
    LastValueSink,
    OpaqueChannel,
    ResultChannel,
    SubscanChannel,
    TeeSink,
)
from .scan_generator import ScanGenerator, ScanOptions, generate_points
from .scan_runner import (
    ScanAxis,
    ScanRunner,
    ScanSpec,
    describe_analyses,
    describe_scan,
    filter_default_analyses,
    select_runner_class,
)
from ..define.utils import dump_json, is_kernel, to_metadata_broadcast_type

__all__ = ["setattr_subscan", "Subscan", "SubscanExpFragment"]

logger = logging.getLogger(__name__)


class Subscan:
    """Handle returned by :meth:`setattr_subscan`, allowing the subscan to actually be
    executed.
    """

    def __init__(
        self,
        runner: ScanRunner,
        fragment: ExpFragment,
        possible_axes: dict[ParamHandle, ScanAxis],
        schema_channel: SubscanChannel,
        coordinate_channels: list[ResultChannel],
        child_result_sinks: dict[ResultChannel, ArraySink],
        flat_dataset_prefix: str,
        flat_segment_start_sink: AppendingDatasetSink,
        aggregate_result_channels: dict[ResultChannel, ResultChannel],
        short_child_channel_names: dict[ResultChannel, str],
        analyses: list[DefaultAnalysis],
        parent_analysis_result_channels: dict[str, ResultChannel],
    ):
        self._runner = runner
        self._fragment = fragment
        self._possible_axes = possible_axes
        self._schema_channel = schema_channel
        self._coordinate_channels = coordinate_channels
        self._child_result_sinks = child_result_sinks
        self._flat_dataset_prefix = flat_dataset_prefix
        self._flat_segment_start_sink = flat_segment_start_sink 
        self._aggregate_result_channels = aggregate_result_channels
        self._short_child_channel_names = short_child_channel_names
        self._analyses = analyses
        self._parent_analysis_result_channels = parent_analysis_result_channels

        # We own the coordinate sinks 
        # (they wouldn't exist without us, as we don't setattr result on some other object
        #  unlike `coordinate_channels` above)
        self._flat_coordinate_sinks = {}
        self._flat_next_point_index = 0
        self._flat_num_segments = 0
        self._flat_current_segment = -1
        self._point_coordinate_sinks = []
        self._flat_site_schema_key = None # Used to ensure reuse consistency


    def set_scan_spec(
        self,
        axis_generators: list[tuple[ParamHandle, ScanGenerator]],
        options: ScanOptions = ScanOptions(),
    ):
        """Bind one concrete runtime scan specification to this subscan.

        At this point, :func:`setup_subscan` has already done the structural work. We
        already know which child fragment is being scanned, which parameter handles are
        allowed to become axes, and which result channels the child fragment can
        produce. We have also already created the parent-owned result channels that
        will later expose the finished subscan.

        What only becomes known here is the runtime scan specification supplied by user
        code: which subset/order of those possible axes will actually be scanned for
        this run, which generators produce the point sequence, and which execution
        options apply. This is enough to build a concrete :class:`ScanSpec`, create the
        temporary sinks that will collect child point coordinates during this run, and
        reset any buffered child-point results from a previous run.

        The actual point values still do not exist here. They are produced later when
        :class:`.ScanRunner` executes the child fragment, first landing in the child-
        side sinks configured here and in :func:`setup_subscan`, and only after the run
        completes are they republished on the parent-owned aggregate result channels.
        """
        axes: list[ScanAxis] = []
        generators: list[ScanGenerator] = []
        # These sinks collect per-point coordinates produced while the child fragment is
        # being scanned. After completion, _push_coordinates() republishes the
        # accumulated arrays into the parent-owned axis result channels created in
        # setup_subscan().
        self._coordinate_sinks = OrderedDict[ParamHandle, ArraySink]()
        self._point_coordinate_sinks = []

        for i, (param_handle, generator) in enumerate(axis_generators):
            axis = self._possible_axes.get(param_handle, None)
            assert axis is not None, "Axis not registered in setattr_subscan()"

            axes.append(axis)
            generators.append(generator)

            # Replicate the TeeSink behaviour to an AppendingDatasetSink
            running_memory_list_sink = ArraySink()
            
            # If they already exist, use existing. This would probably happen if run on
            # this fragment gets called more than once.
            # TODO: We should probably be careful if this gets called by setting up a different 
            # axis scan between runs.
            flat_sink = self._flat_coordinate_sinks.get(i, None)
            if flat_sink is None:
                flat_sink = AppendingDatasetSink(
                    self._runner, self._flat_dataset_prefix+f"points.axis_{i}"
                )
                self._flat_coordinate_sinks[i] = flat_sink

            # The 'this run' corrdinates are only this memory sink that are projected up a level at the end
            self._coordinate_sinks[param_handle] = running_memory_list_sink
            # The live sink that's in the Scan runner path
            self._point_coordinate_sinks.append(TeeSink(running_memory_list_sink, flat_sink))

        # These sinks sit on the child fragment's own result channels and buffer
        # per-point values for the current run until _push_values() republishes them
        # into the parent-owned aggregate result channels.
        had_any = False
        for sink in self._child_result_sinks.values():
            had_any |= len(sink.get_all()) > 0
            sink.clear()
        if had_any:
            # Log to give users a chance to debug continuously interrupted/reset scans
            # (though this may occur as part of normal operation, so even INFO may be
            # too noisy).
            logger.debug("Discarded previous results in Subscan.set_scan_spec()")

        self._spec = ScanSpec(axes, generators, options)
        self._runner.setup(self._fragment, axes,  self._point_coordinate_sinks)
        self._regenerate_points()

    def _regenerate_points(self):
        self._runner.set_points(
            generate_points(self._spec.generators, self._spec.options)
        )

    def run(
        self,
        axis_generators: list[tuple[ParamHandle, ScanGenerator]],
        options: ScanOptions = ScanOptions(),
        execute_default_analyses: bool = True,
    ) -> tuple[
        dict[ParamHandle, list],
        dict[ResultChannel, list],
        dict[str, object],
    ]:
        """Legacy convenience API: configure and execute a complete subscan in one call.

        This is the interface used by :func:`setattr_subscan`. It is primarily a
        host-side convenience wrapper for the older "black box" style of subscan use:
        user code supplies the concrete axis generators/options here, and this method
        takes care of turning them into a runtime scan specification, preparing the
        scanned fragment, executing the scan through :class:`.ScanRunner`, and finally
        republishing the completed subscan on the parent-owned result channels.

        The newer/compositional path is to call :meth:`set_scan_spec` separately and
        then execute via :meth:`acquire` (typically through :class:`SubscanExpFragment`
        and its ``run_once()`` method). In other words:
        - ``run()`` = legacy "configure and execute now"
        - ``acquire()`` = execute an already-configured subscan

        :param axis_generators: The list of scan axes (dimensions). Each element is a
            tuple of parameter to scan (handle must have been passed to
            :func:`setattr_subscan` to set up), and the :class:`ScanGenerator` to use
            to generate the points.
        :param options: :class:`ScanOptions` to control scan execution.
        :param execute_default_analyses: Whether to run any default analyses associated
            with the subfragment after the scan is complete, even if they are not
            exposed as owning fragment channels.

        :return: A tuple ``(coordinates, values, analysis_results)``, each a dictionary
            mapping parameter handles, result channels and analysis channel names to
            lists of their values.
        """
        self.set_scan_spec(axis_generators, options)
        self._prepare_flat_site()
        self._fragment.prepare()
        self._runner.run(
            self._fragment, self._spec, self._point_coordinate_sinks
        )
        return self._push_results(execute_default_analyses)

    @portable
    def acquire(self, execute_default_analyses=False):
        """Execute a subscan that has already been configured via :meth:`set_scan_spec`.

        This is the lower-level execution entry point used by the more modern
        compositional API, in particular :class:`SubscanExpFragment`. Unlike
        :meth:`run`, it does not accept generators/options and does not rebuild the
        scan specification; it assumes that configuration has already happened earlier
        (for example from ``configure()`` during ``host_setup()`` or ``device_setup()``).

        The reason this split exists is partly architectural and partly ARTIQ-related:
        subscan structure can be wired once, concrete scan specs can be updated between
        executions, and on-kernel code needs an execution method that does not bundle
        together all of the host-side configuration work performed by :meth:`run`.
        """
        self._prepare_flat_site()
        if not self._runner.acquire(device_cleanup=False):
            raise RestartKernelTransitoryError("Subscan interrupted by pause request")
        self._finalize(execute_default_analyses)

    @rpc(flags={"async"})
    def _prepare_flat_site(self):
        self._ensure_compatible_flat_site_schema()
        self._broadcast_flat_metadata()
        self._start_flat_segment()

    @rpc(flags={"async"})
    def _finalize(self, execute_default_analyses):
        # Return is ignored for on-kernel-friendly scans.
        self._push_results(execute_default_analyses)
        # Prepare for next subscan.
        self._regenerate_points()

    def _push_results(self, execute_default_analyses):
        analysis_schema, analysis_results = self._handle_default_analyses(
            self._spec.axes, self._coordinate_sinks, execute_default_analyses
        )
        self._push_schema(analysis_schema)
        coordinates = self._push_coordinates()
        values = self._push_values()
        self._finish_flat_segment(coordinates, values)
        return coordinates, values, analysis_results



    def _push_schema(self, analysis_schema):
        scan_schema = describe_scan(
            self._spec, self._fragment, self._short_child_channel_names
        )
        scan_schema.update(analysis_schema)
        self._schema_channel.push(scan_schema)

    def _push_coordinates(self):
        coordinates = OrderedDict()
        for channel, (param, sink) in zip(
            self._coordinate_channels, self._coordinate_sinks.items()
        ):
            v = sink.get_all()
            coordinates[param] = v
            channel.push(v)
            # Prepare for next iteration.
            sink.clear()

        return coordinates

    def _push_values(self):
        values = {}
        for chan, sink in self._child_result_sinks.items():
            v = sink.get_all()
            values[chan] = v
            self._aggregate_result_channels[chan].push(v)
            # Prepare for next iteration.
            sink.clear()
        return values
    
    def _flat_push(self, name: str, value):
        """Push a value into this ScanSite prefix"""
        self._runner.set_dataset(
            self._flat_dataset_prefix+name,
            value,
            broadcast=True
        )

    def _start_flat_segment(self):
        """Mark the current flat-site segment as open, once per configured run."""
        if self._flat_current_segment != -1:
            return

        self._flat_segment_start_sink.push(self._flat_next_point_index)
        self._flat_current_segment = self._flat_num_segments
        self._flat_num_segments += 1
        self._flat_push("current_segment", self._flat_current_segment)

    def _finish_flat_segment(self, coordinates, values):
        self._flat_next_point_index += self._num_points_in_segment(coordinates, values)
        self._flat_current_segment = -1
        self._flat_push("current_segment", -1)

    def _num_points_in_segment(self, coordinates, values) -> int:
        for data in coordinates.values():
            return len(data)
        for data in values.values():
            return len(data)
        return 0

    def _broadcast_flat_metadata(self):
        scan_desc = self._describe_current_scan_without_analysis_results()

        source_prefix = self._runner.get_dataset("system_id", default="rid")
        rid = getattr(self._runner.scheduler, "rid", 0)
        self._flat_push("source_id", f"{source_prefix}_{rid}")

        flat_desc = {
            "fragment_fqn": scan_desc["fragment_fqn"],
            "seed": scan_desc["seed"],
            "axes": scan_desc["axes"],
            "channels": scan_desc["channels"],
            "annotations": scan_desc["annotations"],
            "online_analyses": scan_desc["online_analyses"],
            "analysis_results": {},
            "segment_fields": {"starts": "starts"},
            "segment_state_fields": {"current": "current_segment"},
        }
        for name, value in flat_desc.items():
            ds_value = to_metadata_broadcast_type(value)
            self._flat_push(name, dump_json(value) if ds_value is None else ds_value)

    def _describe_current_scan_without_analysis_results(self):
        def get_axis_index(handle):
            for i, h in enumerate(self._coordinate_sinks.keys()):
                if handle._store == h._store:
                    return i
            assert False

        context = AnnotationContext(
            get_axis_index,
            lambda channel: self._short_child_channel_names[channel],
            lambda channel: False,
        )
        analyses = filter_default_analyses(self._fragment, self._spec.axes)
        scan_desc = describe_scan(
            self._spec, self._fragment, self._short_child_channel_names
        )
        scan_desc.update(describe_analyses(analyses, context))
        scan_desc["analysis_results"] = {}
        return scan_desc
    
    def _current_flat_site_schema_key(self) -> tuple[tuple[str, str], ...]:
        """Return the schema identity for this flat site.

        For now, a flat site is considered compatible if the scanned axes have the same
        identities in the same order. Generator bounds, point counts, and seed may vary
        between runs without requiring a new site.
        """
        return tuple(
            (axis.param_schema["fqn"], axis.path)
            for axis in self._spec.axes
        )

    def _ensure_compatible_flat_site_schema(self) -> None:
        new_key = self._current_flat_site_schema_key()
        if self._flat_site_schema_key is None:
            self._flat_site_schema_key = new_key
            return

        if self._flat_site_schema_key != new_key:
            raise ValueError(
                "Cannot reuse flat subscan site with different axes or axis order"
            )

    def _handle_default_analyses(
        self,
        axes: list[ScanAxis],
        coordinate_sinks: dict[ParamHandle, ArraySink],
        always_run: bool,
    ):
        # Re-filter analyses based on actual scan axes to support slightly dodgy use
        # case where a lower-dimensional scan is actually taken than originally
        # announced – should revisit this design.
        analyses = filter_default_analyses(self._fragment, axes)
        if not analyses:
            return {}, {}

        axis_data = {
            handle._store.identity: sink.get_all()
            for handle, sink in coordinate_sinks.items()
        }

        result_data = {
            chan: sink.get_all() for chan, sink in self._child_result_sinks.items()
        }

        def get_axis_index(handle):
            for i, h in enumerate(coordinate_sinks.keys()):
                if handle._store == h._store:
                    return i
            assert False

        context = AnnotationContext(
            get_axis_index,
            lambda channel: self._short_child_channel_names[channel],
            lambda channel: channel.path in self._parent_analysis_result_channels,
        )
        schema = describe_analyses(analyses, context)
        schema["analysis_results"] = {
            name: parent.path
            for name, parent in self._parent_analysis_result_channels.items()
        }

        analysis_sinks = {}

        if len(self._parent_analysis_result_channels) > 0 or always_run:
            for a in analyses:
                for name, channel in a.get_analysis_results().items():
                    sink = LastValueSink()
                    channel.set_sink(sink)
                    analysis_sinks[name] = sink
            annotations = []
            for a in analyses:
                annotations += a.execute(axis_data, result_data, context)
            if annotations:
                # Replace existing (online-fit) annotations if any analysis produced
                # custom ones. This could be made configurable in the future.
                schema["annotations"] = annotations

        analysis_results = {
            name: sink.get_last() for name, sink in analysis_sinks.items()
        }
        # FIXME: Check for None (not-set) values to produce better error message?
        for name, value in analysis_results.items():
            channel = self._parent_analysis_result_channels.get(name, None)
            if channel is not None:
                channel.push(value)

        return schema, analysis_results


def setattr_subscan(
    owner: Fragment,
    scan_name: str,
    fragment: ExpFragment,
    axis_params: list[tuple[Fragment, str]],
    save_results_by_default: bool = False,
    expose_analysis_results: bool = True,
) -> Subscan:
    """Set up a scan for the given subfragment.

    Result channels are set up in the owning fragment to expose the scan data, such that
    scan results can be inspected after the fact.

    This is the legacy subscan interface, and is geared primarily towards executing the
    scan loop on the host by calling :meth:`Subscan.run` on the returned handle, which
    takes care of setup/results management/etc. all at once. To be able to execute scans
    on-kernel, :class:`.SubscanExpFragment` is preferred, as it directly integrates the
    lifecycle management with the usual setup/cleanup methods, which is more convenient
    in that case.

    :param owner: The fragment to add the subscan to.
    :param scan_name: Name of the scan; appears in result channel names, and the
        :class:`Subscan` instance will be available as ``owner.<scan_name>``.
    :param fragment: The runnable fragment to iterate over in the scan. Must be a
        subfragment of ``owner``.
    :param axis_params: List of `(fragment, param_name)` tuples defining the axes to be
        scanned. It is possible to specify more axes than are actually used; they will
        be overridden and set to their default values.
    :param save_results_by_default: Passed on to all derived result channels.
    :param expose_analysis_results: Whether to add result channels to ``owner`` that
        contain the results of default analyses set for the fragment. Note that for
        this, all results must be known when this function is called (that is, all
        ``axis_params`` should actually be scanned, and the analysis must not fail to
        produce results).

    :return: A :class:`Subscan` instance to use to actually execute the scan.
    """

    assert owner._building, "Can only create a subscan during build_fragment()"
    assert not hasattr(owner, scan_name), f"Field '{scan_name}' already exists"

    # Our own ScanRunner takes care of the fragment lifecycle.
    owner.detach_fragment(fragment)

    subscan = setup_subscan(
        owner,
        f"{scan_name}_",
        fragment,
        axis_params,
        save_results_by_default,
        expose_analysis_results,
    )
    setattr(owner, scan_name, subscan)
    return subscan

def _make_flat_dataset_prefix(
    result_target: Fragment, name_prefix: str, scanned_fragment: ExpFragment
) -> str:
    """Return a simple human-readable dataset prefix for this subscan's flat data.

    The name is derived from:
    - the owning fragment path
    - the subscan name prefix
    - the scanned child fragment FQN
    - the current RID

    This keeps the prefix easy to inspect while developing and is sufficient as a
    first implementation.

    Potential collision:
    this scheme normalises all non-alphanumeric characters to underscores. That means
    distinct fragment paths/FQNs can collapse to the same dataset name. For example,
    ``foo/bar.Baz`` and ``foo.bar/Baz`` both normalise to ``foo_bar_baz``. If such a
    collision occurs within the same RID, two logically distinct subscan sites would
    write to the same flat dataset keys.

    If that becomes a real issue, the usual fix is to append a short stable hash.
    """
    #TODO: This function is overly complicated and naming needs to be fixed. 
    scheduler = result_target.get_device("scheduler")
    rid = getattr(scheduler, "rid", 0)

    owner_str = result_target._stringize_path() or "root"
    scan_name = name_prefix.strip("_") or "subscan"
    child = scanned_fragment.fqn

    site_name = f"{owner_str}_SUBSCAN_{scan_name}_{child}"
    site_name = re.sub(r"[^A-Za-z0-9]+", "_", site_name).strip("_").lower()

    return f"ndscan.rid_{rid}.subscan_flat.{site_name}."



def setup_subscan(
    result_target: Fragment,
    name_prefix: str,
    scanned_fragment: ExpFragment,
    axis_params: list[tuple[Fragment, str]],
    save_results_by_default: bool = False,
    expose_analysis_results: bool = True,
) -> Subscan:
    """Create the internal machinery needed to run a fragment as a subscan.

    From user code, a subscan is meant to behave like a black box owned by the parent:
    running it should look like "execute this child scan and then publish the resulting
    arrays/spec/analysis outputs on parent result channels". Internally, though, that
    black-box behaviour has to be assembled from lower-level pieces.

    This helper is the place where that translation happens. At this point we know the
    structural shape of the subscan, but not yet the concrete runtime scan plan.

    Known here:
    - which child fragment will be scanned
    - which parameter handles are allowed to become scan axes
    - which result channels the child fragment can produce
    - which default-analysis outputs could be exposed on the parent

    Not known here:
    - which subset/order of those possible axes user code will actually scan
    - which generators and options will define the point sequence
    - how many points a particular invocation will contain

    So this function only performs the static wiring. The scanned fragment is detached
    from the parent's normal fragment traversal (before this function was run), fresh
    parameter stores are created for the possible scan axes, the child fragment's own
    result channels are intercepted as the source of per-point values, and new
    parent-owned result channels are created to expose the completed subscan as a
    single parent operation. Any default-analysis outputs that should appear on the
    parent are also registered here, and a runner type compatible with the scanned
    fragment's host/kernel execution mode is instantiated.

    This is also the natural boundary for any live dataset publication. If subscan
    points are mirrored into a flat dataset "scan site" while the scan runs, that
    wiring belongs here as well: the child fragment still looks like a black box to the
    parent, but its per-point coordinates/results can additionally be teed into
    append-only datasets for plotting or later reconstruction.

    Both :func:`setattr_subscan` and :class:`SubscanExpFragment` use this helper so
    that this wiring is defined in one place instead of being duplicated across the two
    public subscan APIs.
    """
    # Override target parameter stores with newly created stores.
    # TODO: Potentially make handles have identity and accept them directly.
    axes = {}
    coordinate_channels = []
    for i, (param_owner, name) in enumerate(axis_params):
        handle = getattr(param_owner, name)
        param, store = param_owner.override_param(name)

        axes[handle] = ScanAxis(
            param.describe(), "/".join(param_owner._fragment_path), store
        )

        # We simply generate sequential result channels to be sure we have enough.
        # Alternatives:
        #  - Require the actually used axes to be given in axis_params (which will be
        #    the most common use case anyway).
        #  - Serialise the scan point coordinates into the scan spec.
        coordinate_channels.append(
            result_target.setattr_result(
                name_prefix + f"axis_{i}",
                OpaqueChannel,
                save_by_default=save_results_by_default,
            )
        )

    # These are the child fragment's own result channels. They remain the source of
    # per-point values during scan execution, even though the completed subscan is
    # republished through result channels owned by result_target.
    original_channels = {}
    scanned_fragment._collect_result_channels(original_channels)

    # TODO: remove name_prefix once naming confusion is sorted
    flat_dataset_prefix = _make_flat_dataset_prefix(result_target, name_prefix, scanned_fragment)

    # Used for indexing the start indices of the subscan data for each subscan run in the flat schema
    # It feels weird here to use `result_target` rather than anything else that `HasEnvironment`. I
    # guess it could be anything?
    flat_segment_start_sink = AppendingDatasetSink(result_target, flat_dataset_prefix+"starts")

    # Short names used when creating parent-owned exports derived from the child
    # channels, e.g. "child/readout/count" -> "count" or ("readout/count" -> "readout_count").
    channel_name_map = shorten_to_unambiguous_suffixes(
        original_channels.keys(), lambda fqn, n: "/".join(fqn.split("/")[-n:])
    )

    # TODO: Make result sinks flat
    # Map child result channels to their temporary in-memory sinks for one subscan run.
    child_result_sinks = {}
    aggregate_result_channels = {}

    # Map child result channels to the short names used in the parent-facing schema.
    short_child_channel_names = {}

    for full_name, short_name in channel_name_map.items():
        original_channel = original_channels[full_name]

        short_identifier = short_name.replace("/", "_")
        short_child_channel_names[original_channel] = short_identifier
        
        # The child fragment still pushes one value per point into its own result
        # channel. We intercept that here so the values can be buffered and later
        # republished on parent-owned aggregate channels.
        # The ArraySink is there as it needs to be cleared each scan run after analysis, for the next run
        running_memory_list_sink = ArraySink()

        if original_channel.save_by_default:
            # However the Dataset sink is there to have live update of all the data produced by this scan
            flat_dataset_sink = AppendingDatasetSink(result_target, flat_dataset_prefix+"points.channel_"+short_identifier)

            # Tee off to both of these Sinks in the machinery
            original_channel.set_sink(TeeSink(running_memory_list_sink, flat_dataset_sink))
        else:
            original_channel.set_sink(running_memory_list_sink)

        # But the branch that does 'this run' stuff only sees the running memory sink
        child_result_sinks[original_channel] = running_memory_list_sink

        # TODO: Implement ArrayChannel to represent a variable number of dimensions
        # around a scalar channel so we can keep the schema information here instead of
        # throwing our hands up in the air helplessly (i.e. using OpaqueChannel).
        # This is the parent-owned export channel for the completed subscan result.
        # TODO: Eventually with flat subscan site dataset schema, I don't think we'll need this? 
        # At least the fact that this will be dataset backed. Perhaps the `setattr` part of this
        # on the parent is important for seeing the result? 
        aggregate_result_channels[original_channel] = result_target.setattr_result(
            name_prefix + "channel_" + short_identifier,
            OpaqueChannel,
            save_by_default=save_results_by_default and original_channel.save_by_default,
        )

    spec_channel = result_target.setattr_result(name_prefix + "spec", SubscanChannel)

    analyses = filter_default_analyses(scanned_fragment, axes.values())
    parent_analysis_result_channels = {}
    if expose_analysis_results:
        analysis_results = reduce(
            lambda x, y: merge_no_duplicates(x, y, kind="analysis result"),
            (a.get_analysis_results() for a in analyses),
            {},
        )
        for name, channel in analysis_results.items():
            # Just clone results channels and directly register them as channels of the
            # owning fragment – perhaps not the cleanest design…
            #
            # TODO: Include "analysis_result" in the full name? Seemed a bit verbose
            # just to avoid collisions in the unlikely case of an analysis result named
            # "spec", "axis_0" or similar.
            full_name = name_prefix + name
            new_channel = copy(channel)
            new_channel.path = "/".join(result_target._fragment_path + [full_name])
            result_target._register_result_channel(
                full_name, new_channel.path, new_channel
            )
            parent_analysis_result_channels[name] = new_channel

    # KLUDGE: If we end up running on the kernel, the ARTIQ compiler needs to treat the
    # "inner" (subscan) and "outer" (TopLevelRunner/…) ScanRunner instances differently
    # in terms of types.
    class RunnerInstance(select_runner_class(scanned_fragment)):
        # KLUDGE: In particular, when we manually set the return type annotations for
        # the parameter value fetching RPC, this should only affect this instance, so
        # override the function. (Would just cloning the function/wrapping it in
        # ScanRunner work?)
        def _get_param_values_chunk(self):
            return super()._get_param_values_chunk()

    runner = RunnerInstance(result_target)

    class SubscanInstance(Subscan):
        # ARTIQ compiler needs a different type for each RunnerInstance.
        pass

    return SubscanInstance(
        runner,
        scanned_fragment,
        axes,
        spec_channel,
        coordinate_channels,
        child_result_sinks,
        flat_dataset_prefix,
        flat_segment_start_sink,
        aggregate_result_channels,
        short_child_channel_names,
        analyses,
        parent_analysis_result_channels,
    )


class SubscanExpFragment(ExpFragment):
    """An :class:`.ExpFragment` that scans another :class:`.ExpFragment` when it
    executes ("subscan").

    Compared to the legacy way of creating subscans, :func:`setattr_subscan`, this
    seamlessly supports the execution of ``@kernel`` subscans: not only can the scanned
    fragment be run on the core device (which the legacy interface supported as well),
    but the :meth:`run_once` method driving the scan itself can also be ``@kernel``.
    This means that :class:`SubscanExpFragment` can be used as part of bigger on-device
    experiments, and that frequent recompilation overhead for repeated subscans can be
    avoided.

    The API of this fragment supports use through composition, which is the natural and
    more flexible way (compared to inheritance). However, when using such a fragment as
    part of a larger code base, be aware of the general restrictions of the ARTIQ
    Python compiler, in particular the fact that all instances of a class must share
    the same type (including attributes, etc.). For this reason, you might want to
    create a separate subtype of this class for each use, such that multiple pieces of
    client code remain composable (can be combined into yet another bigger on-kernel
    program). One way to achieve this is by just creating an "empty" subclass:

    .. code-block:: python

        class Foo(ExpFragment):
            "The fragment to be scanned."
            def build_fragment(self) -> None:
                self.setattr_param("param_a", FloatParam, "a value", default=0.0)
                # […]

            @kernel
            def run_once(self):
                # […]

        class FooSubscan(SubscanExpFragment):
            pass

        class Parent(ExpFragment):
            def build_fragment(self) -> None:
                self.setattr_fragment("foo", Foo)
                self.setattr_fragment("scan", FooSubscan, self, "foo",
                    [(self.foo, "param_a")])
                self.setattr_param("num_scan_points",
                                   IntParam,
                                   "Number of scan points",
                                   default=21,
                                   min=2)

            @rpc(flags={"async"})
            def configure_scan(self):
                if self.num_scan_points.changed_after_use():
                    self.scan.configure([(self.foo.param_a,
                        LinearGenerator(0.0, 0.1, self.num_scan_points.use(),
                                        randomise_order=True))])

            def host_setup(self):
                # Run at least once before kernel starts such that all the fields
                # are initialised (required for the ARTIQ compiler).
                self.configure_scan()
                super().host_setup()

            @kernel
            def device_setup(self):
                # Update scan if num_scan_points was changed (can be left out if
                # there are no scannable parameters influencing the scan settings).
                self.configure_scan()
                self.device_setup_subfragments()

            @kernel
            def run_once(self):
                # Execute the subscan (and anything else that the fragment might
                # need to do).
                self.scan.run_once()

    Another way is to just make the :class:`.ExpFragment` performing the subscan a
    subclass of :class:`SubscanExpFragment`:

    .. code-block:: python

        class Parent(SubscanExpFragment):
            def build_fragment(self) -> None:
                self.setattr_fragment("foo", Foo)
                super().build_fragment(self, "foo", [(self.foo, "param_a")])
                self.setattr_param("num_scan_points",
                                   IntParam,
                                   "Number of scan points",
                                   default=21,
                                   min=2)

            # configure_scan(), host_setup() and device_setup() as above.
    """

    def build_fragment(
        self,
        scanned_fragment_parent: Fragment,
        scanned_fragment: ExpFragment | str,
        axis_params: list[tuple[Fragment, str]],
        save_results_by_default: bool = False,
        expose_analysis_results: bool = True,
    ) -> None:
        """
        :param scanned_fragment_parent: The fragment that owns the scanned fragment.
        :param scanned_fragment: The fragment to scan. Can either be passed as a string
            (the name of the fragment in the parent) or directly as the
            :class:`.ExpFragment` reference.
        :param axis_params: List of `(fragment, param_name)` tuples defining the axes
            to be scanned.
        :param save_results_by_default: Passed on to all derived result channels.
        :param expose_analysis_results: Whether to add result channels to this fragment
            that contain the results of default analyses set for the fragment. Note that
            for this to work, all results must be known when this function is called
            (that is, all ``axis_params`` should actually be scanned, and any analyses
            must not fail to produce results).
        """
        if isinstance(scanned_fragment, str):
            scanned_fragment = getattr(scanned_fragment_parent, scanned_fragment)
        scanned_fragment_parent.detach_fragment(scanned_fragment)
        self._scanned_fragment = scanned_fragment
        # FIXME: Fix subscan model name inference code, remove "_".
        self._subscan = setup_subscan(
            self,
            "_",
            scanned_fragment,
            axis_params,
            save_results_by_default,
            expose_analysis_results,
        )
        if not is_kernel(scanned_fragment.run_once):
            self.run_once = self._subscan.acquire

    def configure(
        self,
        axis_generators: list[tuple[ParamHandle, ScanGenerator]],
        options: ScanOptions = ScanOptions(),
    ) -> None:
        """Configure point generators for each scan axis, and scan options.

        This only needs to be called once (but can be called multiple times to change
        settings between ``run_once()`` invocations, e.g. from a parent fragment
        ``{host, device}_setup()``).

        For on-core-device scans, this has to be called at least once before the kernel
        is first entered (e.g. from ``host_setup()``) such that the types of all the
        fields can be known.

        :param axis_generators: The list of scan axes (dimensions). Each element is a
            tuple of parameter to scan (must correspond to one of the axes specified
            in the constructor; see :meth:`build_fragment`), and the
            :class:`.ScanGenerator` to use to generate the points.
        :param options: :class:`.ScanOptions` to control scan execution.
        """
        self._subscan.set_scan_spec(axis_generators, options)

    # We don't forward prepare(), as there will be a top-level ExpFragment to own the
    # scanned fragment anyway, which can then take care of this directly.

    def host_setup(self):
        """"""
        super().host_setup()
        self._scanned_fragment.host_setup()

    def host_cleanup(self):
        """"""
        self._scanned_fragment.host_cleanup()
        super().host_cleanup()

    # The default device_setup() implementation is appropriate here. It will not forward
    # to the scanned fragment as we have detached the subfragment (but the subscan
    # runner will invoke it before every point).

    @portable
    def device_cleanup(self):
        self._scanned_fragment.device_cleanup()
        self.device_cleanup_subfragments()

    @kernel
    def run_once(self) -> None:
        """Execute the subscan as previously configured.

        This has the usual semantics of a fragment ``run_once()`` method, i.e. calling
        it will acquire one set of results for the fragment (here, a complete scan) and
        write them to the result channels. If the scanned fragment has an ``@kernel``
        ``run_once()`` method, this will automatically be made a ``@kernel`` method as
        well.
        """
        self._subscan.acquire()
