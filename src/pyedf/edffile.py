from tkinter.tix import Tree
from typing import Dict, List, Any, NamedTuple, Tuple
import datetime
from collections import namedtuple


class EDFFile:
    HEADER_LENGTH = 256  # Number of bytes in EDF header

    # Header fields
    VERSION = "version"
    PATIENT_ID = "patient_id"
    RECORDING_ID = "recording_id"
    START_DATE = "start_date"
    START_TIME = "start_time"
    HEADER_BYTES = "header_bytes"
    NUMBER_DATA_RECORDS = "number_data_records"
    DATA_RECORD_DURATION = "data_record_duration"
    NUMBER_SIGNALS = "number_signals"

    SIGNAL_HEADER_LENGTH = 256  # Number of bytes in each signal header

    # Signal header fields
    LABEL = "label"
    TRANSDUCER_TYPE = "transducer_type"
    PHYSICAL_DIMENSION = "physical_dimension"
    PHYSICAL_MIN = "physical_minimum"
    PHYSICAL_MAX = "physical_maximum"
    DIGITAL_MIN = "digital_minimum"
    DIGITAL_MAX = "digital_maximum"
    PREFILTERING = "prefiltering"
    NUMBER_SAMPLES_PER_DATA_RECORD = "number_samples_per_data_record"

    # Shared header fields
    RESERVED = "reserved"

    DATA_SIGNAL_LENGTH = 2  # Number of bytes in a signal
    SIGNAL_EDF_ANNOTATIONS = "EDF Annotations"

    ParsedSample = namedtuple("ParsedSample", ["sample", "time"])
    ParsedAnnotation = namedtuple(
        "ParsedAnnotation", ["onset", "duration", "annotations"])

    def __init__(self, edf_file: str) -> None:
        """ Initialize EDFFile class instance with an EDF file

        Args:
            edf_file (str): Path to EDF file on filesystem
        """
        # Header
        self.version = None
        self.patient_id = None
        self.recording_id = None
        self.start_date = None
        self.start_time = None
        self.header_bytes = None
        self.header_reserved = None
        self.number_data_records = None
        self.data_record_duration = None
        self.number_signals = None

        self.data_records = None

        self.edf_file = edf_file

        self._parse()

    def get_raw_samples_for_signal(self, signal_name: str, record_index: int) -> List[Tuple[int, int]]:
        """ Get samples as two bytes for a signal

        Args:
            signal_name (str): Signal name
            record_index (int): Signal record index

        Returns:
            List[Tuple[int, int]]: List of tuples samples
        """
        samples = self.data_records[record_index].get(signal_name)
        return samples

    def get_parsed_samples_for_signal(self, signal_name: str, record_index: int) -> List[Dict[str, Any]]:
        """ Return samples parsed as integers and signal gain applied, with the time of each sample

        Args:
            signal_name (str): Signal name
            record_index (int): Signal record index

        Returns:
            List[Dict[str, Any]]: List of samples
        """
        signal_duration_ms = (self.data_record_duration /
                              self.signal_header.get(signal_name).get(
                                  "number_samples_per_data_record")) * 1000
        signal_gain = self.signal_header.get(signal_name).get("signal_gain")

        raw_samples = self.get_raw_samples_for_signal(
            signal_name, record_index)

        if signal_name == EDFFile.SIGNAL_EDF_ANNOTATIONS:
            raw_annotation = bytearray(
                [sample for sample_pair in raw_samples for sample in sample_pair]).decode("UTF-8")
            parsed_samples = self._parse_annotation(raw_annotation)
        else:
            # Convert samples to integers and apply the signal gain
            int_samples = [int.from_bytes(
                s, "little", signed=True) * signal_gain for s in raw_samples]

            parsed_samples = []
            for i, sample in enumerate(int_samples):
                # Calculate the sample time
                sample_datetime = datetime.datetime.combine(self.start_date, self.start_time) + \
                    datetime.timedelta(seconds=self.data_record_duration*record_index) + \
                    datetime.timedelta(milliseconds=i*signal_duration_ms)
                parsed_samples.append(EDFFile.ParsedSample(
                    sample=sample, time=sample_datetime))

        return parsed_samples

    def get_parsed_recording_id(self) -> Dict[str, str]:
        """ Parse the recording ID field into individual data points

        Raises:
            Exception: Raised if raw recording ID is not formatted correctly

        Returns:
            Dict[str, str]: Parsed recording ID data 
        """
        parts = self.recording_id.split(" ")
        if len(parts) < 5:
            raise Exception("Invalid recording ID, not enough data")
        if parts[0] != "Startdate":
            raise Exception(
                "Invalid recording ID. Does not begin with 'Startdate'.")

        parsed = {
            "start_date": parts[1],
            "hospital_admin_code": parts[2],
            "responsible_investgator": parts[3],
            "used_equipment": parts[4]
        }

        if len(parts) > 5:
            parsed["remaining_data"] = " ".join(parts[5:])

        return parsed

    def _parse_annotation(self, annotation: str) -> Tuple[int, int, List[str]]:
        list_of_timestamp_annotation_lists: List[str] = annotation.rstrip(
            "\x00").split("\x00")

        parsed_annotations: List[EDFFile.ParsedAnnotation] = []

        for tal in list_of_timestamp_annotation_lists:
            tal_parts: List[str] = tal.rstrip("\x14").split("\x14")
            str_onset: str = None
            str_duration: str = None
            try:
                str_onset, str_duration = tal_parts[0].split("\x15")
            except ValueError:
                str_onset = tal_parts[0]

            annotations: List[str] = None
            if len(tal_parts) > 1:
                annotations = tal_parts[1].split("\x14")

            onset: datetime.datetime = datetime.datetime.combine(
                self.start_date, self.start_time) + datetime.timedelta(seconds=int(str_onset))
            try:
                duration: datetime.timedelta = datetime.timedelta(
                    seconds=int(str_duration))
            except TypeError:
                duration = None

            parsed_annotation = EDFFile.ParsedAnnotation(
                onset, duration, annotations)
            parsed_annotations.append(parsed_annotation)

        return parsed_annotations

    def _parse_header(self, data: bytes):
        """ Parse the EDF header fields

        Args:
            data (bytes): Data bytes from EDF file
        """
        self.version = int(data[0:8].decode().rstrip())
        self.patient_id = data[8:88].decode().rstrip()
        self.recording_id = data[88:168].decode().rstrip()
        self.start_date = datetime.datetime.strptime(
            data[168:176].decode().rstrip(), "%d.%m.%y").date()
        self.start_time = datetime.datetime.strptime(
            data[176:184].decode().rstrip(), "%H.%M.%S").time()
        self.header_bytes = int(data[184:192].decode().rstrip())
        self.header_reserved = data[192:236].decode().rstrip()
        self.number_data_records = int(data[236:244].decode().rstrip())
        self.data_record_duration = float(data[244:252].decode().rstrip())
        self.number_signals = int(data[252:256].decode().rstrip())

    def _parse_signal_header(self, data: bytes, number_signals: int) -> Dict[str, Dict]:
        """ Parse the signals header

        Args:
            data (bytes): Data byes from EDF file
            number_signals (int): Number of signals to parse (learned from header)

        Returns:
            Dict[str, Dict]: Dictionary of signal specifications. Keys are signal names and values describe the signal.
        """
        signals = []

        signals = [{EDFFile.LABEL: data[0+(i*16):16+(i*16)].decode().rstrip()}
                   for i in range(0, number_signals)]
        data = data[number_signals*16:]

        [signals[i].update({EDFFile.TRANSDUCER_TYPE: data[0+(i*80):80+(i*80)].decode().rstrip()})
         for i in range(0, number_signals)]
        data = data[number_signals*80:]

        [signals[i].update({EDFFile.PHYSICAL_DIMENSION: data[0+(i*8):8+(i*8)].decode().rstrip()})
         for i in range(0, number_signals)]
        data = data[number_signals*8:]

        [signals[i].update({EDFFile.PHYSICAL_MIN: float(
            data[0+(i*8):8+(i*8)].decode().rstrip())}) for i in range(0, number_signals)]
        data = data[number_signals*8:]

        [signals[i].update({EDFFile.PHYSICAL_MAX: float(
            data[0+(i*8):8+(i*8)].decode().rstrip())}) for i in range(0, number_signals)]
        data = data[number_signals*8:]

        [signals[i].update({EDFFile.DIGITAL_MIN: int(
            data[0+(i*8):8+(i*8)].decode().rstrip())}) for i in range(0, number_signals)]
        data = data[number_signals*8:]

        [signals[i].update({EDFFile.DIGITAL_MAX: int(
            data[0+(i*8):8+(i*8)].decode().rstrip())}) for i in range(0, number_signals)]
        data = data[number_signals*8:]

        [signals[i].update({EDFFile.PREFILTERING: data[0+(i*80):80+(i*80)].decode().rstrip()})
         for i in range(0, number_signals)]
        data = data[number_signals*80:]

        [signals[i].update({EDFFile.NUMBER_SAMPLES_PER_DATA_RECORD: int(
            data[0+(i*8):8+(i*8)].decode().rstrip())}) for i in range(0, number_signals)]
        data = data[number_signals*8:]

        [signals[i].update({EDFFile.RESERVED: data[0+(i*32):32+(i*32)].decode().rstrip()})
         for i in range(0, number_signals)]

        [signals[i].update({"signal_gain":
                            (signals[i].get(EDFFile.PHYSICAL_MAX) - signals[i].get(EDFFile.PHYSICAL_MIN)) /
                            (signals[i].get(EDFFile.DIGITAL_MAX) -
                             signals[i].get(EDFFile.DIGITAL_MIN))
                            }) for i in range(0, number_signals)]

        signals_dict = {signal[EDFFile.LABEL]: signal for signal in signals}
        return signals_dict

    def _parse(self):
        """ Read the EDF file into memory and parse it
        """
        with open(self.edf_file, "rb") as file:
            header_data = file.read(EDFFile.HEADER_LENGTH)
            self._parse_header(header_data)

            signal_header_data = file.read(
                EDFFile.SIGNAL_HEADER_LENGTH * self.number_signals)
            self.signal_header = self._parse_signal_header(
                signal_header_data, self.number_signals)

            self.data_records = []
            for x in range(self.number_data_records):
                data_record = {}
                for signal in self.signal_header.values():
                    samples = file.read(
                        signal[EDFFile.NUMBER_SAMPLES_PER_DATA_RECORD] * EDFFile.DATA_SIGNAL_LENGTH)

                    data_record.update({signal[EDFFile.LABEL]: [tuple(
                        samples[i:i+2]) for i in range(0, len(samples), 2)]})

                self.data_records.append(data_record)


class ResmedEDFFile(EDFFile):
    def get_parsed_samples_for_signal(self, signal_name: str, record_index: int):
        """ Get samples for a signal. Parses sample based on Resmed implementation.

        Args:
            signal_name (str): Signal name
            record_index (int): Signal index

        Returns:
            List[int]: List of samples
        """
        samples = super().get_parsed_samples_for_signal(signal_name, record_index)

        if signal_name == "Date":
            # Raw data is days since January 1, 1970, let's convert it to a datetime.datetime
            samples = list(map(lambda ps: EDFFile.ParsedSample(sample=ps.sample, time=datetime.datetime(1970, 1, 1) + datetime.timedelta(
                days=ps.sample)), samples))
        elif signal_name in ("MaskOn", "MaskOff"):
            # Raw value is number of minutes since 12:00 on Date, let's convert it to a datetime.datetime
            signal_date = (self.get_parsed_samples_for_signal("Date", record_index)[
                           0].time).replace(hour=12, minute=0, second=0)
            samples = list(map(lambda ps: EDFFile.ParsedSample(sample=ps.sample, time=signal_date + datetime.timedelta(minutes=ps.sample))
                           if ps.sample != -1 else ps, samples))

        return samples

    def get_parsed_recording_id(self) -> Dict[str, str]:
        """ Parse the recording ID field into individual data points

        Raises:
            Exception: Raised if raw recording ID is not formatted correctly

        Returns:
            Dict[str, str]: Parsed recording ID data 
        """

        edf_parsed = super().get_parsed_recording_id()

        resmed_parts = edf_parsed.pop("remaining_data", "").split(" ")
        if len(resmed_parts) < 1:
            return edf_parsed

        resmed_parsed = {part.split("=")[0]: part.split(
            "=")[1] for part in resmed_parts if "=" in part}

        return {**edf_parsed, **resmed_parsed}
