// Toggle sidebar on mobile
function toggleSidebar() {
       document.getElementById('sidebar').classList.toggle('show');
   }
   
   // Toggle profile menu
   function toggleProfileMenu() {
       document.getElementById('profileMenu').classList.toggle('show');
   }
   
   // Create new conversation - Redirect to home page
   function createNewConversation() {
       window.location.href = '/';
   }
   
   // Auto-resize textarea
   function autoResize(textarea) {
       textarea.style.height = 'auto';
       textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
   }
   
   // Load specific conversation
   function loadConversation(conversationId) {
       window.location.href = `/load_conversation/${conversationId}`;
   }
   
   function submitQuestion() {
       const form = document.getElementById('questionForm');
       if (form.checkValidity()) {
           document.getElementById('loader').style.display = 'block';
           const sendBtn = document.querySelector('.send-btn');
           sendBtn.disabled = true;
           sendBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
           form.submit();
       }
   }
   
   function scrollToBottom() {
       const chatContainer = document.getElementById('chatContainer');
       if (chatContainer) {
           chatContainer.scrollTop = chatContainer.scrollHeight;
       }
   }
   
   function handleKeyPress(event) {
       if (event.key === 'Enter' && !event.shiftKey) {
           event.preventDefault();
           submitQuestion();
       }
   }
   
   function convertToIndianTime(utcDateString) {
       const date = new Date(utcDateString);
       const indianTime = new Date(date.getTime() + (5.5 * 60 * 60 * 1000));
       const now = new Date();
       const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
       const messageDate = new Date(indianTime.getFullYear(), indianTime.getMonth(), indianTime.getDate());
       
       const diffTime = today - messageDate;
       const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
       
       let dateLabel = '';
       if (diffDays === 0) {
           dateLabel = 'Today';
       } else if (diffDays === 1) {
           dateLabel = 'Yesterday';
       } else {
           dateLabel = indianTime.toLocaleDateString('en-IN', {
               day: '2-digit',
               month: 'short',
               year: 'numeric'
           });
       }
       
       const timeStr = indianTime.toLocaleTimeString('en-IN', { 
           hour: '2-digit', 
           minute: '2-digit',
           hour12: false 
       });
       
       return `${dateLabel}, ${timeStr}`;
   }
   
   function confirmDelete(event) {
       if (!confirm('Are you sure you want to delete this conversation?')) {
           event.preventDefault();
           return false;
       }
       return true;
   }
   
   document.addEventListener('click', function(event) {
       const profileMenu = document.getElementById('profileMenu');
       const userProfile = document.querySelector('.user-profile');
       
       if (!userProfile.contains(event.target) && !profileMenu.contains(event.target)) {
           profileMenu.classList.remove('show');
       }
   });
   
   document.addEventListener('DOMContentLoaded', function() {
       scrollToBottom();
       
       const textarea = document.querySelector('textarea[name="question"]');
       if (textarea) {
           textarea.focus();
           textarea.addEventListener('keypress', handleKeyPress);
       }
       
       const timeElements = document.querySelectorAll('.message-time');
       timeElements.forEach(element => {
           const utcTime = element.getAttribute('data-utc');
           if (utcTime) {
               element.textContent = convertToIndianTime(utcTime);
           }
       });
   });