import React, { Component } from 'react'
import Stream from './Stream';


 class Feed extends Component {
  state = {
    srcs: [
      'http://0.0.0.0:5000/video_feed?si=0',
      'http://0.0.0.0:5000/video_feed?si=1',
      'http://0.0.0.0:5000/video_feed?si=2',
      'http://0.0.0.0:5000/video_feed?si=3',

    ]
  };


  render() {
    return (
      
        <div className="feed">
          <Stream  src ={this.state.srcs[0]} class = 'stream' />
          <Stream src ={this.state.srcs[1]}  class = 'stream' />
          <Stream src ={this.state.srcs[2]}  class = 'stream' />
          <Stream src ={this.state.srcs[3]}  class = 'stream' />
   
          
        </div>
      
    )
  }
}


export default Feed;
