from typing import List, Union, Optional

import matplotlib.backend_bases as back
import matplotlib.pyplot as plt
import numpy as np

from labtools.utils.convenience import limit_to_range


class CuttingTool:
    """
    A class to cut signals into pieces. The pieces can be of equal or different lengths.
    The number of pieces must be provided.

    Parameters:
        num_pieces (int): Number of resulting pieces.
        mode (str): "start_stop", "separate", or "same_length" - mode of cutting.
            - If "start_stop", the pieces are cut by specifying the start and end indices of each piece.
            - If "separate", the pieces are cut by specifying the start index of each piece.
            - If "same_length", num_pieces pieces of length length_pieces_ms are cut from a single start point.
        length_pieces_ms (int or List[int], optional): Length of the resulting pieces in milliseconds.
            - If int, all pieces have the same length.
            - If a list, each element is the length of the corresponding piece.
    """

    def __init__(
            self,
            num_pieces: int,
            mode: str = "start_stop",
            length_pieces_ms: Optional[Union[int, List[int]]] = None,
    ):
        self.num_pieces = num_pieces
        self.mode = mode
        self.length_pieces_ms = length_pieces_ms
        self.validate_input_types()
        self.initialize_cuts()

    @property
    def num_cuts(self) -> int:
        return self._num_cuts

    def initialize_cuts(self):
        if self.mode == "start_stop":
            self._num_cuts = 2 * self.num_pieces
        elif self.mode == "separate":
            self._num_cuts = self.num_pieces - 1
        elif self.mode == "same_length":
            self._num_cuts = 1  # User specifies only the start point
            if not isinstance(self.length_pieces_ms, int):
                raise ValueError(
                    "length_pieces_ms must be an integer when mode is 'same_length'"
                )
        else:
            raise ValueError(
                f"mode must be 'start_stop', 'separate', or 'same_length', not '{self.mode}'"
            )

        if isinstance(self.length_pieces_ms, list):
            if len(self.length_pieces_ms) != self.num_pieces:
                raise ValueError(
                    f"length_pieces_ms must be a list of {self.num_pieces} integers"
                )

    def validate_input_types(self):
        if not isinstance(self.num_pieces, int):
            raise TypeError("num_pieces must be an integer")
        if not isinstance(self.mode, str):
            raise TypeError("mode must be a string")
        if self.mode not in {"start_stop", "separate", "same_length"}:
            raise ValueError('mode must be "start_stop", "separate", or "same_length"')
        if self.length_pieces_ms is not None:
            if not isinstance(self.length_pieces_ms, (int, list)):
                raise TypeError(
                    "length_pieces_ms must be an integer or a list of integers"
                )
            if isinstance(self.length_pieces_ms, list):
                if not all(isinstance(x, int) for x in self.length_pieces_ms):
                    raise TypeError(
                        "All elements in length_pieces_ms must be integers"
                    )

    def cut(self, data):
        if isinstance(self.length_pieces_ms, int):
            # Cut into pieces of a specified length
            return [
                data[i: i + self.length_pieces_ms]
                for i in range(0, len(data), self.length_pieces_ms)
            ]
        elif isinstance(self.length_pieces_ms, list):
            # Cut into pieces of varying lengths
            pieces = []
            start = 0
            for length in self.length_pieces_ms:
                pieces.append(data[start: start + length])
                start += length
            return pieces
        else:
            # Cut into equal sized pieces based on number of pieces
            cut_size = len(data) // self.num_pieces
            return [
                data[i * cut_size: (i + 1) * cut_size] for i in range(self.num_pieces)
            ]

    @staticmethod
    def _process_start_stop(cut_marks: List[int]) -> List[tuple]:
        return [(start, end) for start, end in zip(cut_marks[::2], cut_marks[1::2])]

    @staticmethod
    def _process_separate(cut_marks: List[int]) -> List[tuple]:
        extended_cuts = [0] + cut_marks + [-1]
        return [
            (extended_cuts[i], extended_cuts[i + 1])
            for i in range(len(extended_cuts) - 1)
        ]

    def _process_same_length(self, cut_marks: List[int]) -> List[tuple]:
        if not isinstance(self.length_pieces_ms, int):
            raise ValueError(
                "length_pieces_ms must be an integer when mode is 'same_length'"
            )
        start_point = cut_marks[0]
        return [
            (start_point + i * self.length_pieces_ms, start_point + (i + 1) * self.length_pieces_ms)
            for i in range(self.num_pieces)
        ]

    def make_tuples(self, cut_marks: List[int]) -> List[tuple]:
        cut_marks = sorted(cut_marks)
        if self.mode == "start_stop":
            return self._process_start_stop(cut_marks)
        elif self.mode == "separate":
            return self._process_separate(cut_marks)
        elif self.mode == "same_length":
            return self._process_same_length(cut_marks)
        else:
            raise ValueError(f"mode '{self.mode}' not recognized")

    @staticmethod
    def cut_array(
            array: np.ndarray, cut_marks: List[tuple], axis: int = 0
    ) -> List[np.ndarray]:
        return [array[start:end] for start, end in cut_marks]

    def cut_dictionary(
            self, data_in: dict, cut_marks: List[tuple], axis: int = 0
    ) -> List[dict]:
        result = []
        for start, end in cut_marks:
            piece = {}
            for k, v in data_in.items():
                piece[k] = self.cut_array(v, [(start, end)], axis=axis)[0]
            result.append(piece)
        return result


class InteractiveCuttingTool(CuttingTool):
    """
    An interactive cutting tool that allows users to select cut points via a plotted interface.

    Parameters:
        num_pieces (int): Number of resulting pieces.
        mode (str): "start_stop", "separate", or "same_length" - mode of cutting.
        length_pieces_ms (int or List[int], optional): Length of the resulting pieces in milliseconds.
        plot_title (str, optional): Title of the plot.
    """

    def __init__(
            self,
            num_pieces: int,
            mode: str = "start_stop",
            length_pieces_ms: Optional[Union[int, List[int]]] = None,
            plot_title: str = "",
    ):
        super().__init__(num_pieces, mode, length_pieces_ms)
        self.plot_title = plot_title

    def set_plot_title(self, title: str):
        self.plot_title = title

    def add_info_text(self, ax):
        x = ax.get_xlim()[1] * 0.98
        y = ax.get_ylim()[1] * 0.98
        return ax.text(
            x,
            y,
            f"You can use all features with the left button (zoom, and move....) \n"
            f"The right button allows you to select the cut location \n Make {self.num_cuts} cuts",
            fontsize=12,
            color="black",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
            horizontalalignment="right",
            verticalalignment="top",
        )

    @staticmethod
    def add_done_text(ax):
        x = ax.get_xlim()[1] * 0.02
        y = ax.get_ylim()[1] * 0.98
        return ax.text(
            x,
            y,
            "You can close the window now",
            fontsize=12,
            color="black",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
            verticalalignment="top",
        )

    @staticmethod
    def add_cut_line(ax, x: int):
        return ax.axvline(x, color="k", linestyle="--")

    def get_cut_marks(
            self,
            display_data: np.ndarray,
            timestamp_ms: Optional[np.ndarray] = None,
            axis: int = 0,
    ) -> List[int]:
        # Plot the data array
        fig = plt.figure("Original data")
        fig.suptitle(self.plot_title)
        ax = fig.add_subplot(111)

        if timestamp_ms is None:
            timestamp_ms = np.arange(display_data.shape[axis])

        ax.plot(timestamp_ms, display_data)
        # Optionally plot resultant magnitude of 3D data
        if (display_data.ndim == 2) and (3 in display_data.shape):
            a = display_data.shape.index(3)
            resultant = np.linalg.norm(display_data, axis=a)
            ax.plot(timestamp_ms, resultant, "k--", linewidth=0.5)
        info_text = self.add_info_text(ax)
        cut_marks = []
        for _ in range(self.num_cuts):
            pts = fig.ginput(
                1, timeout=0, mouse_add=back.MouseButton.RIGHT, mouse_pop=None
            )
            if not pts:
                continue  # Skip if no point was selected
            new_mark = limit_to_range(
                int(pts[0][0]), int(timestamp_ms[0]), int(timestamp_ms[-1])
            )
            self.add_cut_line(ax, new_mark)
            cut_marks.append(new_mark)
        if self.mode == "same_length":
            if not isinstance(self.length_pieces_ms, int):
                raise ValueError(
                    "length_pieces_ms must be an integer when mode is 'same_length'"
                )
            preview_marks = [
                cut_marks[0] + i * self.length_pieces_ms
                for i in range(1, self.num_pieces + 1)
            ]
            for mark in preview_marks:
                self.add_cut_line(ax, mark)
                cut_marks.append(mark)
        info_text.remove()
        self.add_done_text(ax)
        plt.show()
        plt.close()

        # Efficient index finding with np.searchsorted
        cut_marks_index = np.searchsorted(timestamp_ms, cut_marks)
        return cut_marks_index.tolist()

    def plot_and_cut(
            self,
            data_in: Union[dict, np.ndarray],
            axis: int = 0,
            display_key: Optional[str] = None,
            **kwargs,
    ) -> List[Union[np.ndarray, dict]]:
        """
        Plot the data and get cut marks from user input.

        Parameters:
            data_in (dict or np.ndarray): Data to be cut. Can be a dictionary of arrays or a single array.
                Arrays in the dictionary must be of the same length.
            axis (int): Axis along which to cut.
            display_key (str, optional): Key of the dictionary that is used to display the data for getting the cut marks.
            **kwargs: Additional keyword arguments to be passed to get_cut_marks.

        Returns:
            List of cut pieces (dictionaries or arrays).
        """
        if isinstance(data_in, dict):
            if display_key is None:
                raise ValueError(
                    "display_key must be provided when data_in is a dictionary"
                )
            data = data_in[display_key]
            timestamp = data_in.get("timestamp")
        else:
            data = data_in
            timestamp = None

        if timestamp is None:
            # If timestamp is not provided, create one based on data length
            timestamp = np.arange(data.shape[axis])

        ts = timestamp.copy()
        cut_marks = self.get_cut_marks(data, timestamp_ms=ts, axis=axis)
        cut_mark_tuples = self.make_tuples(cut_marks)
        if isinstance(data_in, dict):
            return self.cut_dictionary(data_in, cut_mark_tuples, axis=axis)
        else:
            return self.cut_array(data_in, cut_mark_tuples, axis=axis)
