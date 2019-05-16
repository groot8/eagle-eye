import React, { Component } from 'react'
import './Sidebar.css';
import axios from 'axios';

 class Sidebar extends Component{
  state = {
    list_ids : []
  }
  getlist(){
    return axios.get('http://0.0.0.0:5000/ids').then(r => r.data).catch(()=>[])
  }
  componentDidMount(){
    setInterval(() => {
      this.getlist().then(list_ids => {
        this.setState(()=>{
          return {
            list_ids,
            disabled: false
          }
        })
      })
    }, 2000);
  }
  toggleVis(id){
    if (this.state.disabled) return
    var update = (disabled)=>{
      this.setState(()=>{
        return {
          disabled
        }
      })
    }
    update(true)
    axios.get('http://0.0.0.0:5000/ids/'+(id[1] ? 'show': 'hide')+'?id='+id[0]).then(()=>{
      update(false)
    }).catch(()=>{
      update(false)
    })
  }
  render(){
    return (
      <div className='side-bar'>
      <h3>List of Trackers</h3>
        {this.state.list_ids.map(id => (<div className={`person ${id[1]?'red':'green'} ${this.state.disabled?'disabled':''}` } key={id[0]} onClick={this.toggleVis.bind(this,id)}> person {id[0]}</div>))}
      </div>
    )
  }
}

export default Sidebar;
