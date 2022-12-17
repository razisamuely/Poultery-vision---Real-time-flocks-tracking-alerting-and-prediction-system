from datetime import datetime, timedelta
import os
import hikload.hikvisionapi as hikvisionapi
import time
from hikload.download import search_for_recordings, download_recordings
import pytz
from get_configs_as_class import getConfigs
from typing import List
import logging

class Recording():
    def __init__(self, cid, cname, url, startTime,endTime):
        self.cid = cid
        self.cname = cname
        self.url = url
        self.startTime = startTime
        self.endTime = endTime

def search_for_recordings(server: hikvisionapi.HikvisionServer, args) -> List[Recording]:
    channelList = server.Streaming.getChannels()
    channelids = []
    channels = []
    if args.photos:
        for channel in channelList['StreamingChannelList']['StreamingChannel']:
            if (int(channel['id']) % 10 == 1) and (args.cameras is None or channel['id'] in args.cameras):
                # Force looking at the hidden 103 channel for the photos
                channel['id'] = str(int(channel['id'])+2)
                channelids.append(channel['id'])
                channels.append(channel)
        logging.info("Found channels %s" % channelids)
    else:
        for channel in channelList['StreamingChannelList']['StreamingChannel']:
            if (int(channel['id']) % 10 == 1) and (args.cameras is None or channel['id'] in args.cameras):
                channelids.append(channel['id'])
                channels.append(channel)
        logging.info("Found channels %s" % channelids)

    downloadQueue = []
    for channel in channels:
        cname = channel['channelName']
        cid = channel['id']
        if args.days:
            endtime = datetime.now().replace(
                hour=23, minute=59, second=59, microsecond=0)
            starttime = endtime - timedelta(days=args.days)
        elif args.yesterday:
            endtime = datetime.now().replace(
                hour=23, minute=59, second=59, microsecond=0) - timedelta(days=1)
            starttime = endtime - timedelta(days=1)
        else:
            starttime = args.starttime
            endtime = args.endtime

        logging.debug("Using %s and %s as start and end times" %
                      (starttime.isoformat() + "Z", endtime.isoformat() + "Z"))

        try:
            if args.allrecordings:
                recordings = server.ContentMgmt.search.getAllRecordingsForID(
                    cid)
                logging.info("There are %s recordings in total for channel %s" %
                             (recordings['CMSearchResult']['numOfMatches'], cid))
            else:
                recordings = server.ContentMgmt.search.getPastRecordingsForID(
                    cid, starttime.isoformat() + "Z", endtime.isoformat() + "Z")
                logging.info("Found %s recordings for channel %s" %
                             (recordings['CMSearchResult']['numOfMatches'], cid))
        except hikvisionapi.classes.HikvisionException:
            logging.error("Could not get recordings for channel %s" % cid)
            continue

        # This loops from every recording
        if recordings['CMSearchResult']['numOfMatches'] != "0":
            recordinglist = recordings['CMSearchResult']['matchList']['searchMatchItem']
        else:
            recordinglist = []
        # In case there is only one recording, we need to make it a list
        if type(recordinglist) is not list:
            recordinglist = [recordinglist]
        result = []
        for i in recordinglist:
            result.append(Recording(
                cid=cid,
                cname=cname,
                url=i['mediaSegmentDescriptor']['playbackURI'],
                startTime=i['timeSpan']['startTime'],
                endTime=i['timeSpan']['endTime']
                ))
            logging.debug("Found recording type %s on channel %s" % (
                i['mediaSegmentDescriptor']['contentType'], cid
            ))
            if not args.photos and i['mediaSegmentDescriptor']['contentType'] != 'video':
                # This recording is not a video, skip it
                continue
        downloadQueue.extend(result)
    return downloadQueue

class dirToClass(object):
    pass


class getVideosFromNVR(object):
    """
    Get videos from NVR  with respect to local time zone.
    """
    def __init__(self, videos_configs_path: str, zone: str = "Israel"):
        self._videos_configs_path = videos_configs_path
        self.videos_configs = getConfigs().get_configs_as_class(configs_path=videos_configs_path)
        self.args = getConfigs().get_configs_as_class(data=self.videos_configs.hikload_args)
        self._country_time_zone = pytz.timezone(zone)

    def update_last_video_downloaded_datetime(self, last_video_time_path: str, time: datetime):
        # 1.1 If file is exist:
        if not os.path.exists(last_video_time_path):
            os.system(f"touch {last_video_time_path}")

        with open(last_video_time_path, 'w') as f:
            f.write(time)

    def define_start_end_time(self, args, run_from_last_video_time: bool):
        """
        Defining start and end time to start reading recordings from nvr
        If no times supllied over the configs file, start from now minus minute
        """

        if run_from_last_video_time:
            with open(args.last_video_list_path, "r") as f:
                last_time = f.read()

            args.starttime = datetime.strptime(last_time, "%Y-%m-%dT%H:%M:%SZ")
            args.endtime = args.starttime + timedelta(hours=12)


        elif args.starttime is None:
            args.starttime = datetime.now(self._country_time_zone) - timedelta(minutes=300)
            args.endtime = datetime.now(self._country_time_zone) + timedelta(hours=12)

        else:
            args.starttime = datetime.strptime(args.starttime, "%Y-%m-%d %H:%M:%S")
            args.endtime = datetime.strptime(args.endtime, "%Y-%m-%d %H:%M:%S")
        return args

    def connect_to_server(self, args):
        """
        Create server object for getting recordings list and downloading recordings
        """
        server = hikvisionapi.HikvisionServer(host=args.host,
                                              user=args.user,
                                              password=args.password)
        return server

    def search_for_recordings_new(self, server: hikvisionapi, args, number_of_tries: int) :
        """
        Get all recordings from start and end time interval.
        Dropg (downloadQueue[1:-1]) tiles of not completed last and first record and c
        After number_of_tries is exceeded error is raised.

        """
        print("starttime = ",args.starttime)
        print("endtime = ", args.endtime)
        for i in range(number_of_tries):
            try:
                downloadQueue = search_for_recordings(server=server,
                                                      args=args)

                break

            except Exception as e:
                if i < number_of_tries:
                    print(f"{datetime.utcnow().strftime('%H:%M:%S')}--- Warning, failed to search_for_recordings, "
                          f"try number = {i}, taking sleep of 2 ")
                    time.sleep(2)
                else:
                    raise e
        downloadQueue = sorted(downloadQueue, key=lambda x: x.startTime)
        print('downloadQueue = ', downloadQueue)
        last_video_time = datetime.strptime(downloadQueue[-1].startTime, "%Y-%m-%dT%H:%M:%SZ")
        downloadQueue = downloadQueue[1:-1]
        return downloadQueue, last_video_time

    def wait_for_ready_recordings(self, last_video_time: datetime):
        """
        Calculate sleeping time for record to ready for download
        """
        #  Wait for 2 minutes diff
        diff = (datetime.now(self._country_time_zone).replace(tzinfo=None) - last_video_time).seconds
        print(f"{self._country_time_zone} time now = ", datetime.now(self._country_time_zone).replace(tzinfo=None))
        print("last_video_time = ", last_video_time)
        videos_diffs = 120
        if diff < videos_diffs:
            print(f"""{datetime.utcnow().strftime("%H:%M:%S")} -- Sleeping {videos_diffs - diff + 2} seconds""")
            time.sleep(videos_diffs - diff + 2)

    def update_start_end_time(self, args, downloadQueue):
        """
        Update start and end time with respect to last downloaded record
        """
        args.starttime = datetime.strptime(downloadQueue[-1].startTime, "%Y-%m-%dT%H:%M:%SZ") + timedelta(seconds=1)
        args.endtime = args.starttime + timedelta(hours=12)
        return args

    def download_recordings(self, recordingobj: hikvisionapi, server: hikvisionapi, args, number_of_tries: int):
        """
        Try number_of_tries do download record
        Create files if needed, cd to child file, and go back to parent file.
        """
        for i in range(number_of_tries):
            try:
                print(
                    f"""\n\n{datetime.utcnow().strftime("%H:%M:%S")}  --- Started download record from NVR ---\nrecords time = {recordingobj.startTime}""")
                original_path = os.path.abspath(os.getcwd())

                download_recordings(
                    server=server,
                    args=args,
                    downloadQueue=[recordingobj]
                )
                break

            except Exception as e:
                # if i < number_of_tries:
                #     print(
                #         f"{datetime.utcnow().strftime('%H:%M:%S')} = Warning, faile to download video, try number = {i}, takin zz of 2 seconds")
                #     time.sleep(2)
                # else:
                    raise e

            finally:
                os.chdir(original_path)
