import React from "react";


export type VideoProps = {
  src: string
}

const Video : React.FC<VideoProps> = ({ src }) => {
  return (
    <video controls width="100%" preload="auto" >
      <source src={src} type="video/mp4"></source>
      Your browser does not support HTML5 video.
    </video>
  )
}

export default Video;

